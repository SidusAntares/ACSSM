import torch
import torch.nn.functional as F
from torch import nn


class ClassCenterAligner(nn.Module):
    """类中心对齐模块：对齐源域和目标域的类中心"""

    def __init__(self, num_classes, feature_dim, device,
                 momentum=0.9, alignment_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.alignment_weight = alignment_weight
        self.device = device

        # ✅ 初始化源域和目标域的类中心
        self.register_buffer('src_centers',
                             torch.zeros(num_classes, feature_dim, device=device))
        self.register_buffer('trg_centers',
                             torch.zeros(num_classes, feature_dim, device=device))
        self.register_buffer('src_counts',
                             torch.zeros(num_classes, device=device))
        self.register_buffer('trg_counts',
                             torch.zeros(num_classes, device=device))

        # # ✅ 调试信息
        # print(f"✅ ClassCenterAligner初始化:")
        # print(f"   num_classes={num_classes}, feature_dim={feature_dim}")
        # print(f"   src_centers shape: {self.src_centers.shape}")
        # print(f"   trg_centers shape: {self.trg_centers.shape}")

    def update_src_centers(self, features, labels):
        """
        更新源域类中心（使用移动平均）

        Args:
            features: [B, C] 或 [B, T, C]
            labels: [B] 或 [B, T]
        """
        # ✅ 处理时间维度
        if features.dim() == 3:  # [B, T, C]
            B, T, C = features.shape
            # print(f"  [Aligner] 输入 features: [B={B}, T={T}, C={C}]")

            # 压缩时间维度：取平均
            features = features.mean(dim=1)  # [B, C]

            # 处理标签
            if labels.dim() == 2:  # [B, T]
                labels = labels[:, 0]  # 取第一个时间步的标签 [B]

        # 确保labels是一维的
        if labels.dim() > 1:
            labels = labels.squeeze()

        B, C = features.shape
        # print(f"  [Aligner] 处理后 features: [B={B}, C={C}], labels: {labels.shape}")
        # print(f"  [Aligner] src_centers shape: {self.src_centers.shape}")

        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                class_features = features[mask]  # [N, C]
                batch_center = class_features.mean(0)  # [C]
                count = mask.sum().float()

                # print(f"    类别 {c}: 样本数={class_features.size(0)}, batch_center shape={batch_center.shape}")

                # ✅ 移动平均更新
                alpha = self.momentum
                # self.src_centers[c] 形状应该是 [C]
                self.src_centers[c] = alpha * self.src_centers[c] + (1 - alpha) * batch_center
                self.src_counts[c] += count

    def update_trg_centers(self, features, pseudo_labels, confidence=None):
        """
        更新目标域类中心（可选：加权更新）

        Args:
            features: [B, C] 或 [B, T, C]
            pseudo_labels: [B] 或 [B, T]
            confidence: [B] 或 [B, T] (可选)
        """
        # ✅ 处理时间维度
        if features.dim() == 3:  # [B, T, C]
            B, T, C = features.shape
            # print(f"  [Aligner] 输入 features: [B={B}, T={T}, C={C}]")

            # 压缩时间维度：取平均
            features = features.mean(dim=1)  # [B, C]

            # 处理标签
            if pseudo_labels.dim() == 2:  # [B, T]
                pseudo_labels = pseudo_labels[:, 0]  # [B]

            # 处理置信度
            if confidence is not None and confidence.dim() == 2:
                confidence = confidence[:, 0]  # [B]

        # 确保labels是一维的
        if pseudo_labels.dim() > 1:
            pseudo_labels = pseudo_labels.squeeze()

        B, C = features.shape
        # print(f"  [Aligner] 处理后 features: [B={B}, C={C}], pseudo_labels: {pseudo_labels.shape}")
        # print(f"  [Aligner] trg_centers shape: {self.trg_centers.shape}")

        for c in range(self.num_classes):
            mask = (pseudo_labels == c)
            if mask.any():
                class_features = features[mask]  # [N, C]

                # 可选：根据置信度加权
                if confidence is not None:
                    class_conf = confidence[mask]
                    batch_center = (class_features * class_conf.unsqueeze(1)).sum(0) / class_conf.sum()
                else:
                    batch_center = class_features.mean(0)  # [C]

                count = mask.sum().float()
                alpha = self.momentum
                # self.trg_centers[c] 形状应该是 [C]
                self.trg_centers[c] = alpha * self.trg_centers[c] + (1 - alpha) * batch_center
                self.trg_counts[c] += count

    def compute_alignment_loss(self, method='mse'):
        """计算类中心对齐损失"""
        valid_mask = (self.src_counts > 0) & (self.trg_counts > 0)
        src_valid = torch.norm(self.src_centers, dim=1) > 1e-6
        trg_valid = torch.norm(self.trg_centers, dim=1) > 1e-6
        valid_mask = valid_mask & src_valid & trg_valid

        if valid_mask.sum() == 0:
            return torch.tensor(0.0).to(self.device)

        if method == 'mse':
            loss = F.mse_loss(
                self.src_centers[valid_mask],
                self.trg_centers[valid_mask]
            )
        elif method == 'cosine':
            # 余弦相似度对齐（鼓励类中心方向一致）
            src_norm = F.normalize(self.src_centers[valid_mask], dim=1)
            trg_norm = F.normalize(self.trg_centers[valid_mask], dim=1)
            loss = 1 - (src_norm * trg_norm).sum(dim=1).mean()
        else:
            raise ValueError(f"Unknown alignment method: {method}")

        return self.alignment_weight * loss

    def reset(self):
        """重置类中心（用于新epoch或新阶段）"""
        self.src_centers.zero_()
        self.trg_centers.zero_()
        self.src_counts.zero_()
        self.trg_counts.zero_()

    def get_centers(self):
        """获取当前类中心（用于可视化或调试）"""
        return {
            'src_centers': self.src_centers.detach().cpu(),
            'trg_centers': self.trg_centers.detach().cpu(),
            'src_counts': self.src_counts.detach().cpu(),
            'trg_counts': self.trg_counts.detach().cpu()
        }


class TFMPTF_Projector(nn.Module):
    """
    可训练的投影头（放在ACSSM中注册）
    负责将TFMPTF的原始特征投影到低维空间
    """

    def __init__(self, args, input_tmptm_dim=576, input_fmptm_dim=6,
                 output_dim=128, num_groups=32):
        super().__init__()
        self.num_groups = num_groups
        self.output_dim = output_dim

        # 时域投影：576 → 64
        self.time_projector = nn.Sequential(
            nn.Linear(input_tmptm_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim // 4),  # 576 → 32
            nn.LayerNorm(output_dim // 4)
        )

        # 频域投影：6 → 64
        self.freq_projector = nn.Sequential(
            nn.Linear(input_fmptm_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim // 4),  # 6 → 32
            nn.LayerNorm(output_dim // 4)
        )

        # 组级别注意力融合
        self.group_attention = nn.MultiheadAttention(
            embed_dim=output_dim // 2,  # 64
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 全局投影：64 → 128
        self.global_projector = nn.Sequential(
            nn.Linear(output_dim // 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, tfmptf_features):
        """
        Args:
            tfmptf_features: dict {
                'group_tmptm': [B, num_groups, 576]
                'group_fmptm': [B, num_groups, 6]
            }

        Returns:
            time_features: [B, output_dim]  # 时域全局特征
            freq_features: [B, output_dim]  # 频域全局特征
            group_features: [B, num_groups, output_dim//2]  # 组级别特征
        """
        group_tmptm = tfmptf_features['group_tmptm']  # [B, 32, 576]
        group_fmptm = tfmptf_features['group_fmptm']  # [B, 32, 6]

        # 投影到低维
        group_tmptm_proj = self.time_projector(group_tmptm)  # [B, 32, 32]
        group_fmptm_proj = self.freq_projector(group_fmptm)  # [B, 32, 32]

        # 合并时频特征
        group_combined = torch.cat([group_tmptm_proj, group_fmptm_proj], dim=2)  # [B, 32, 64]

        # 组级别注意力融合
        group_attended, _ = self.group_attention(
            group_combined, group_combined, group_combined
        )  # [B, 32, 64]

        global_features = group_attended.mean(dim=1)  # [B, 64]
        global_proj = self.global_projector(global_features)  # [B, 128]

        # 直接返回全局特征，不需要拆分
        return global_proj, global_proj, group_attended


class AttentionGate(nn.Module):
    """时频特征注意力融合门控"""

    def __init__(self, feature_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** -0.5
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, time_feat, freq_feat):
        """
        Args:
            time_feat: [B, D] - 时域特征
            freq_feat: [B, D] - 频域特征

        Returns:
            fused: [B, D] - 融合后的特征
        """
        # 拼接时频特征: [B, 2, D]
        combined = torch.stack([time_feat, freq_feat], dim=1)

        # 自注意力计算
        Q = self.query(combined)  # [B, 2, D]
        K = self.key(combined)  # [B, 2, D]
        V = self.value(combined)  # [B, 2, D]

        # 计算注意力权重: [B, 2, 2]
        # Q @ K^T 得到 [B, 2, 2]，表示每个位置对其他位置的注意力
        attn_weights = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)

        # 加权求和: [B, 2, D]
        # 每个位置的输出是所有位置的加权和
        attended = attn_weights @ V

        # 融合两个头的输出
        # 使用第一个位置的注意力权重来融合两个头
        # attn_weights[:, 0, :] 形状: [B, 2]，表示位置0对两个头的注意力
        fusion_weights = attn_weights[:, 0, :].unsqueeze(-1)  # [B, 2, 1]

        # 加权融合: [B, 2, D] * [B, 2, 1] -> [B, 2, D] -> sum(dim=1) -> [B, D]
        fused = (attended * fusion_weights).sum(dim=1)

        # 可选：投影层增强表达能力
        fused = self.output_proj(fused)

        return fused


class GLUAttentionGate(nn.Module):
    """基于GLU的高效门控 - 速度提升3-4倍"""

    def __init__(self, feature_dim):
        super().__init__()
        hidden_dim = feature_dim * 2

        # GLU结构
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2)
        )

        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, time_feat, freq_feat):
        # 拼接输入
        combined = torch.cat([time_feat, freq_feat], dim=-1)  # [B, 2D]

        # GLU门控
        gate_output = self.gate_net(combined)  # [B, 2D]
        gate, transform = gate_output.chunk(2, dim=-1)  # 各 [B, D]

        # 门控融合
        gate = torch.sigmoid(gate)
        fused = gate * time_feat + (1 - gate) * transform  # [B, D]

        fused = self.output_proj(fused)
        return fused