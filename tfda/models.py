import torch
import torch.nn as nn

import torch
import torch.nn as nn

class TEncoder(nn.Module):
    def __init__(self, configs):
        super(TEncoder, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = self.aap(x).view(x.shape[0], -1)

        return x_flat, x

class FEncoder(nn.Module):
    def __init__(self, hparams):
        super(FEncoder, self).__init__()
        self.hparams = hparams

class AdEncoder(nn.Module):
    """
    适配版 TFDA 特征编码器骨架

    设计原则:
        1. 模块化：将时域、频域、融合、投影拆分为独立组件。
        2. 动态适配：通过 tfda_instance 获取输入尺寸，拒绝硬编码。
        3. 职责单一：每个子模块只负责一种数据处理。
    """

    def __init__(self, args, matrix_size, state_dim, matrix_num, fmptm_len):
        super(AdEncoder, self).__init__()

        self.matrix_size = matrix_size
        self.state_dim = state_dim
        self.matrix_num = matrix_num
        self.fmptm_len = fmptm_len

        # ==========================================
        # 组件 1: 时域矩阵编码器 (TEncoder)
        # ==========================================
        # 职责: 处理 TMPTM 矩阵，提取空间结构特征
        # 对应原始 TFDA 中的: self.cnn (Time Feature Encoder)
        # 这里的实现将替换为 Conv2d 结构
        self.TEncoder = TEncoder(args)

        # ==========================================
        # 组件 2: 频域向量编码器 (FEncoder)
        # ==========================================
        # 职责: 处理 FMPTM 向量，提取模态相关性特征
        # 对应原始 TFDA 中的: self.freq_feature + self.cnn_f
        # 这里的实现将替换为 Conv1d/Linear 结构
        self.FEncoder = FEncoder(args)

        # ==========================================
        # 组件 3: 特征融合器 (Fuser)
        # ==========================================
        # 职责: 将时域和频域特征进行拼接和维度对齐
        # 对应原始 TFDA 中的: 隐式的 torch.cat 操作
        self.fuser = self._build_fuser()

        # ==========================================
        # 组件 4: 投影头 (Projector)
        # ==========================================
        # 职责: 映射到高维空间，用于对抗训练或对比学习
        # 对应原始 TFDA 中的: self.projector
        # 保持原始 TFDA 的 MLP 结构不变
        feat_dim = args.final_out_channels * args.features_len
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feat_dim)
        )

    def _build_fuser(self):
        """
        构建特征融合器
        TODO: 填充特征拼接或加权逻辑
        """
        return nn.Identity()  # 暂时使用恒等映射


    def forward(self, x):
        """
        前向传播骨架

        Args:
            x: [Batch, Input_Channels, Matrix_Size, Matrix_Size]
               来自 TFMPTF 的输出
        """
        # 1. 拆解输入
        # TODO: 将 x 拆分为 tmptm (矩阵) 和 fmptm (向量)

        # 2. 并行提取
        # time_feat = self.matrix_encoder(tmptm)
        # freq_feat = self.vector_encoder(fmptm)

        # 3. 融合
        # fused_feat = self.fuser(time_feat, freq_feat)

        # 4. 投影
        # z = self.projector(fused_feat)

        return None, None  # 返回占位符