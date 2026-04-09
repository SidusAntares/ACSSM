import torch
import torch.nn as nn
import numpy as np
import math
from scipy.signal import windows

class TFMPTF(nn.Module):
    """
    优化版 TFMPTF
    优化点:
        1. 向量化 _compute_tmptm，去除 Python 时间循环。
        2. VMD 使用高斯窗平滑滤波，减少频谱泄露。
        3. FMPTM 仅提取上三角特征，减少冗余。
    """

    def __init__(self, args):
        super(TFMPTF, self).__init__()
        self.state_dim = args.state_dim
        self.vmd_modes = args.vmd_modes
        self.perm_dim = args.perm_dim
        self.num_groups = args.num_groups
        assert self.state_dim % self.num_groups == 0
        self.state = self.state_dim // self.num_groups

        self.matrix_num = 2

        # 预计算阶乘，避免重复计算
        self.max_patterns = math.factorial(self.perm_dim)
        self.tmptm_dim = self.max_patterns ** 2  # 576
        self.fmptm_dim = self.vmd_modes * (self.vmd_modes - 1) // 2
        self.fact_cache = [math.factorial(i) for i in range(self.perm_dim)]

        # 预计算排列模式的所有可能排序 (加速 Lehmer Code 计算)
        # shape: [max_patterns, perm_dim]
        all_perms = np.array(list(self._generate_permutations(self.perm_dim)))
        self.register_buffer('perm_table', torch.tensor(all_perms, dtype=torch.long))
        self.register_buffer("matrix_size", torch.tensor(self.max_patterns, dtype=torch.int))


    def _generate_permutations(self, m):
        """递归生成全排列"""
        if m == 1:
            yield (0,)
        else:
            for p in self._generate_permutations(m - 1):
                for i in range(m):
                    yield p[:i] + (m - 1,) + p[i:]

    def _get_permutation_pattern_fast(self, windows_data):
        """
        向量化计算 Lehmer Code
        Args:
            windows_data: np.array, shape [Num_Windows, m]
        Returns:
            ids: np.array, shape [Num_Windows]
        """
        sort_indices = np.argsort(windows_data, axis=1)

        ids = np.zeros(windows_data.shape[0], dtype=int)

        for i in range(self.perm_dim):
            if i < self.perm_dim - 1:
                smaller_counts = np.sum(sort_indices[:, i:i+1] > sort_indices[:, i+1:], axis=1)
                ids += smaller_counts * self.fact_cache[self.perm_dim - 1 - i]

        return ids

    def _vmd_decomposition(self, signal):
        """
        改进版 VMD (基于高斯窗的频域滤波)
        """
        K = self.vmd_modes
        N = len(signal)
        if N < K * 2:
            return np.tile(signal, (K, 1))

        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N)

        modes = np.zeros((K, N))

        # 使用高斯窗构建平滑的带通滤波器
        # 宽度参数控制模态的带宽
        bandwidth = 1.0 / K

        for k in range(K):
            center_freq = (k - K / 2) / K  # 中心化频率分布

            dist = np.abs(freqs - center_freq)

            mask = np.exp(-0.5 * (dist / bandwidth) ** 2)

            # 归一化能量 (可选)
            # mask /= np.sum(mask)

            filtered_fft = fft_signal * mask
            mode_k = np.real(np.fft.ifft(filtered_fft))
            modes[k, :] = mode_k

        return modes

    def _compute_tmptm(self, modes):
        """
        向量化版 TMPTM 计算
        """
        K, time_len = modes.shape
        m = self.perm_dim
        max_patterns = self.max_patterns

        tmptm = np.zeros((max_patterns, max_patterns))

        # 1. 构建滑动窗口矩阵 (使用 stride_tricks 避免内存复制)
        # shape: [K, Num_Windows, m]
        shape = (K, time_len - m + 1, m)
        strides = (modes.strides[0], modes.strides[1], modes.strides[1])
        windows_matrix = np.lib.stride_tricks.as_strided(modes, shape=shape, strides=strides)

        # 2. 批量计算所有窗口、所有模态的 Pattern ID
        # 重塑为 [K * Num_Windows, m] 进行计算
        flat_windows = windows_matrix.reshape(-1, m)
        flat_ids = self._get_permutation_pattern_fast(flat_windows)

        # 3. 计算转移
        num_windows = flat_ids.shape[0] // K
        flat_ids = flat_ids.reshape(K, num_windows)

        for k in range(K):
            ids = flat_ids[k]
            # 当前时刻 t 和下一时刻 t+1
            curr_ids = ids[:-1]
            next_ids = ids[1:]

            # 使用 numpy 的 bincount 进行快速直方图统计
            # 将二维坐标 (row, col) 映射到一维 index = row * N + col
            linear_indices = curr_ids * max_patterns + next_ids
            counts = np.bincount(linear_indices, minlength=max_patterns**2)

            tmptm += counts.reshape(max_patterns, max_patterns)

        # 4. 归一化
        row_sums = tmptm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return tmptm / row_sums

    def _compute_fmptm(self, modes):
        """
        精简版 FMPTM (只取上三角)
        """
        K = modes.shape[0]
        energies = modes ** 2

        try:
            corr_matrix = np.corrcoef(energies)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        except:
            corr_matrix = np.eye(K)

        # 只提取上三角部分 (不含对角线，因为对角线恒为1，无信息量)
        # 或者包含对角线，视具体需求而定
        triu_indices = np.triu_indices(K, k=1)
        return corr_matrix[triu_indices]

    def forward(self, hidden_states):
        batch_size, time_steps, state_dim = hidden_states.shape

        # 分离计算，避免内存爆炸
        group_tmptm_list = []
        group_fmptm_list = []

        # 分组处理
        for g in range(self.num_groups):
            start_idx = g * self.state
            end_idx = start_idx + self.state
            group_data = hidden_states[:, :, start_idx:end_idx]  # [B, T, 16]

            # 组内平均池化（在state维度）
            group_pooled = group_data.mean(dim=2)  # [B, T]

            batch_tmptm = []
            batch_fmptm = []

            # 对每个样本计算TFMPTF特征
            for b in range(batch_size):
                signal = group_pooled[b].detach().cpu().numpy()  # [T]

                # VMD分解
                modes = self._vmd_decomposition(signal)  # [vmd_modes, T]

                # 计算TMPTM
                tmptm = self._compute_tmptm(modes).flatten()  # [576]

                # 计算FMPTM
                fmptm_vec = self._compute_fmptm(modes)  # [6]

                batch_tmptm.append(tmptm)
                batch_fmptm.append(fmptm_vec)

            # 转换为tensor
            tmptm_tensor = torch.tensor(np.array(batch_tmptm),
                                        dtype=torch.float32).to(hidden_states.device)
            fmptm_tensor = torch.tensor(np.array(batch_fmptm),
                                        dtype=torch.float32).to(hidden_states.device)

            group_tmptm_list.append(tmptm_tensor.unsqueeze(1))  # [B, 1, 576]
            group_fmptm_list.append(fmptm_tensor.unsqueeze(1))  # [B, 1, 6]

        # 拼接所有组
        group_tmptm = torch.cat(group_tmptm_list, dim=1)  # [B, 32, 576]
        group_fmptm = torch.cat(group_fmptm_list, dim=1)  # [B, 32, 6]

        return {
            'group_tmptm': group_tmptm,  # [B, 32, 576]
            'group_fmptm': group_fmptm  # [B, 32, 6]
        }
