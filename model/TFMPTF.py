import torch
import torch.nn as nn
import torch.fft as fft
import math


class TFMPTF(nn.Module):
    """
    GPU优化版 TFMPTF - 完全在GPU上运行
    修复类型转换错误和混合精度问题
    """

    def __init__(self, args):
        super(TFMPTF, self).__init__()
        self.state_dim = args.state_dim
        self.vmd_modes = args.vmd_modes
        self.perm_dim = args.perm_dim
        self.num_groups = args.num_groups
        assert self.state_dim % self.num_groups == 0
        self.state_per_group = self.state_dim // self.num_groups

        # 预计算参数
        self.max_patterns = math.factorial(self.perm_dim)
        self.tmptm_dim = self.max_patterns ** 2
        self.fmptm_dim = self.vmd_modes * (self.vmd_modes - 1) // 2

        # 修复1：将fact_cache改为整数类型
        self.register_buffer('fact_cache',
                             torch.tensor([math.factorial(i) for i in range(self.perm_dim)],
                                          dtype=torch.long))  # long类型

        # 预计算窗口索引（用于滑动窗口）
        self.register_buffer('window_indices',
                             torch.arange(self.perm_dim).unsqueeze(0))

    def _vmd_decomposition_gpu(self, signals):
        """
        GPU向量化VMD分解
        Args:
            signals: [batch_size, num_groups, time_steps]
        Returns:
            modes: [batch_size, num_groups, vmd_modes, time_steps]
        """
        batch_size, num_groups, T = signals.shape
        K = self.vmd_modes

        # FFT变换
        fft_signal = fft.fft(signals, dim=2)  # [B, G, T]

        # 创建频率轴
        freqs = torch.fft.fftfreq(T, device=signals.device, dtype=torch.float32)  # [T]

        # 创建高斯滤波器矩阵 [K, T]
        bandwidth = 1.0 / K
        center_freqs = torch.linspace(-0.5, 0.5, K, device=signals.device, dtype=torch.float32)  # [K]

        # 计算所有模态的滤波器
        # [K, 1] - [1, T] -> [K, T]
        dist = (freqs.unsqueeze(0) - center_freqs.unsqueeze(1)).abs()
        filters = torch.exp(-0.5 * (dist / bandwidth) ** 2)  # [K, T]

        # 应用滤波器并进行逆FFT
        # [B, G, 1, T] * [1, 1, K, T] -> [B, G, K, T]
        filtered_fft = fft_signal.unsqueeze(2) * filters.unsqueeze(0).unsqueeze(0)
        modes = fft.ifft(filtered_fft, dim=3).real  # [B, G, K, T]

        return modes

    def _compute_pattern_ids_vectorized(self, windows_data):
        """
        向量化计算排列模式ID（Lehmer Code）
        Args:
            windows_data: [batch_size, num_groups, K, num_windows, perm_dim]
        Returns:
            pattern_ids: [batch_size, num_groups, K, num_windows]
        """
        B, G, K, N, m = windows_data.shape

        # 排序获取索引 [B, G, K, N, m]
        sorted_indices = windows_data.argsort(dim=4)

        # 计算Lehmer Code
        # 对于每个位置，计算右侧比它小的元素个数
        pattern_ids = torch.zeros(B, G, K, N, device=windows_data.device, dtype=torch.long)

        for i in range(m - 1):
            # 当前位置的索引 [B, G, K, N, 1]
            curr_idx = sorted_indices[..., i:i + 1]
            # 右侧所有位置的索引 [B, G, K, N, m-i-1]
            right_indices = sorted_indices[..., i + 1:]

            # 计算比当前元素小的数量
            smaller_counts = (curr_idx > right_indices).sum(dim=4)  # [B, G, K, N]

            # 修复2：使用in-place加法，确保类型一致
            pattern_ids = pattern_ids + smaller_counts * self.fact_cache[m - 1 - i]

        return pattern_ids

    def _compute_tmptm_gpu(self, modes):
        """
        GPU向量化TMPTM计算
        Args:
            modes: [batch_size, num_groups, vmd_modes, time_steps]
        Returns:
            tmptm: [batch_size, num_groups, max_patterns^2]
        """
        B, G, K, T = modes.shape
        m = self.perm_dim
        max_patterns = self.max_patterns

        # 创建滑动窗口
        # 使用unfold创建窗口 [B, G, K, num_windows, m]
        num_windows = T - m + 1
        windows = modes.unfold(3, m, 1)  # [B, G, K, num_windows, m]

        # 计算所有窗口的pattern ID
        pattern_ids = self._compute_pattern_ids_vectorized(windows)  # [B, G, K, num_windows]

        # 计算转移矩阵
        # 当前时刻和下一时刻的pattern ID
        curr_ids = pattern_ids[..., :-1]  # [B, G, K, num_windows-1]
        next_ids = pattern_ids[..., 1:]  # [B, G, K, num_windows-1]

        # 将二维索引映射到一维
        # [B, G, K, N] -> [B, G, K, N]
        linear_indices = curr_ids * max_patterns + next_ids

        # 使用one_hot和求和来统计转移次数
        # [B, G, K, N, max_patterns^2]
        one_hot = torch.nn.functional.one_hot(linear_indices, num_classes=max_patterns ** 2)
        tmptm = one_hot.sum(dim=3).float()  # [B, G, K, max_patterns^2]

        # 对所有模态求和
        tmptm = tmptm.sum(dim=2)  # [B, G, max_patterns^2]

        # 归一化
        row_sums = tmptm.sum(dim=2, keepdim=True)  # [B, G, 1]
        row_sums = row_sums.clamp(min=1.0)  # 避免除零
        tmptm = tmptm / row_sums

        return tmptm

    def _compute_fmptm_gpu(self, modes):
        """
        GPU向量化FMPTM计算
        Args:
            modes: [batch_size, num_groups, vmd_modes, time_steps]
        Returns:
            fmptm: [batch_size, num_groups, fmptm_dim]
        """
        B, G, K, T = modes.shape

        # 计算能量
        energies = modes ** 2  # [B, G, K, T]

        # 标准化
        mean_energy = energies.mean(dim=3, keepdim=True)  # [B, G, K, 1]
        std_energy = energies.std(dim=3, keepdim=True)  # [B, G, K, 1]
        std_energy = std_energy.clamp(min=1e-8)
        normalized_energies = (energies - mean_energy) / std_energy  # [B, G, K, T]

        # 计算相关系数矩阵
        # [B, G, K, T] @ [B, G, T, K] -> [B, G, K, K]
        corr_matrix = torch.matmul(normalized_energies, normalized_energies.transpose(2, 3)) / T

        # 提取上三角部分（不含对角线）
        triu_mask = torch.triu(torch.ones(K, K, device=modes.device), diagonal=1).bool()

        # 使用mask提取上三角
        fmptm = corr_matrix[:, :, triu_mask].reshape(B, G, -1)  # [B, G, fmptm_dim]

        return fmptm

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, time_steps, state_dim]
        Returns:
            dict with tmptm and fmptm features (float32)
        """
        batch_size, time_steps, state_dim = hidden_states.shape

        # 修复3：确保输入是float32
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()

        # 重新排列为 [batch_size, num_groups, state_per_group, time_steps]
        hidden_states = hidden_states.permute(0, 2, 1)  # [B, state_dim, T]
        hidden_states = hidden_states.reshape(batch_size, self.num_groups, self.state_per_group, time_steps)

        # 组内平均池化 [B, G, T]
        group_pooled = hidden_states.mean(dim=2)

        # VMD分解 [B, G, K, T]
        modes = self._vmd_decomposition_gpu(group_pooled)

        # 修复4：移除autocast，直接计算并确保输出为float32
        # TFMPTF模块主要做FFT和统计计算，对精度要求较高，不需要混合精度
        tmptm = self._compute_tmptm_gpu(modes)  # [B, G, 576]
        fmptm = self._compute_fmptm_gpu(modes)  # [B, G, 6]

        # 确保输出是float32
        tmptm = tmptm.float()
        fmptm = fmptm.float()

        return {
            'group_tmptm': tmptm,  # [batch_size, num_groups, tmptm_dim]
            'group_fmptm': fmptm  # [batch_size, num_groups, fmptm_dim]
        }