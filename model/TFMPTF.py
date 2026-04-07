import torch
import torch.nn as nn
import numpy as np
import math
from scipy.signal import windows

class TFMPTF_Optimized(nn.Module):
    """
    优化版 TFMPTF
    优化点:
        1. 向量化 _compute_tmptm，去除 Python 时间循环。
        2. VMD 使用高斯窗平滑滤波，减少频谱泄露。
        3. FMPTM 仅提取上三角特征，减少冗余。
    """

    def __init__(self, args):
        super(TFMPTF_Optimized, self).__init__()
        self.state_dim = args.state_dim
        self.vmd_modes = args.vmd_modes
        self.perm_dim = args.perm_dim
        self.time_steps = args.T

        # 预计算阶乘，避免重复计算
        self.max_patterns = math.factorial(self.perm_dim)
        self.fact_cache = [math.factorial(i) for i in range(self.perm_dim)]

        # 预计算排列模式的所有可能排序 (加速 Lehmer Code 计算)
        # shape: [max_patterns, perm_dim]
        all_perms = np.array(list(self._generate_permutations(self.perm_dim)))
        self.register_buffer('perm_table', torch.tensor(all_perms, dtype=torch.long))

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
        # 获取排序索引: argsort 返回的是每行从小到大的索引
        # shape: [Num_Windows, m]
        sort_indices = np.argsort(windows_data, axis=1)

        ids = np.zeros(windows_data.shape[0], dtype=int)

        # 计算逆序数 (Lehmer Code)
        # 这里虽然还是循环 m，但 m 通常很小 (3, 4, 5)，外层的大循环已被向量化
        for i in range(self.perm_dim):
            # 统计后面有多少个比当前小
            # sort_indices[:, i:i+1] shape: [N, 1]
            # sort_indices[:, i+1:]   shape: [N, m-i-1]
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

            # 高斯函数: exp(-0.5 * ((f - fc) / bw)^2)
            # 注意处理周期性频率 (虽然对于低频模态影响不大)
            dist = np.abs(freqs - center_freq)
            # 考虑负频率的周期性距离 (简化处理，仅针对主频带)
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
        batch_size = hidden_states.shape[0]
        all_features = []

        # 为了演示，这里保持 Batch 循环 (Batch 较小时不可避免)
        for i in range(batch_size):
            sample_feats = []
            signal_data = hidden_states[i].detach().cpu().numpy()

            for d in range(self.state_dim):
                signal = signal_data[:, d]
                modes = self._vmd_decomposition(signal)

                tmptm = self._compute_tmptm(modes)
                fmptm_vec = self._compute_fmptm(modes)

                # 策略调整：
                # 1. TMPTM 也可以只取上三角或特定统计量，防止维度过大
                # 这里假设 max_patterns=24，24*24=576 维，尚可接受
                sample_feats.append(tmptm.flatten())
                sample_feats.append(fmptm_vec)

            # 拼接所有通道
            final_vec = np.concatenate(sample_feats)
            all_features.append(final_vec)

        return torch.tensor(np.array(all_features), dtype=torch.float32).to(hidden_states.device)