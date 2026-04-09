import torch
import torch.nn as nn
import torch.nn.functional as F

from tfda.models import AdEncoder


class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反转层 (Gradient Reversal Layer, GRL)

    职责:
        在前向传播中保持输入不变。
        在反向传播中将梯度乘以负系数 -lambda。
        这是实现域对抗训练的核心组件。
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 梯度反转：乘以 -lambda
        grad_input = grad_output.neg() * ctx.lambda_
        return grad_input, None


class TFDA(nn.Module):
    """
    Time-Frequency Domain Adaptation (TFDA) Module

    职责:
        接收 TFMPTF 输出的矩阵特征，进行特征降维、分类预测和域判别。
        通过对抗训练学习域不变特征。
    """

    def __init__(self, tfmptf, args):
        super(TFDA, self).__init__()

        # --- 输入配置 ---
        self.matrix_size = tfmptf.matrix_size.item()  # TFMPTF 输出矩阵的大小
        self.matrix_num = tfmptf.matrix_num.item()  # 通道数 (TMPTM + FMPTM)
        self.input_channels = tfmptf.state_dim * self.matrix_num
        self.state_dim = tfmptf.state_dim
        self.fmptm_len = tfmptf.fmptm_len

        # --- 特征头 (Feature Head) ---
        # 职责: 将二维矩阵特征展平并映射为高维向量
        # 使用卷积层提取局部特征，然后展平
        self.feature_head = AdEncoder(args, self.matrix_size, self.state_dim, self.matrix_num, self.fmptm_len)

        # --- 分类器 (Classifier) ---
        # 职责: 预测样本的类别标签 (用于源域监督)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, args.num_classes)
        )

        # --- 域判别器 (Discriminator) ---
        # 职责: 预测样本属于源域还是目标域 (用于对抗训练)
        self.discriminator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出 0-1 之间的概率
        )

    def forward(self, mtf_features, lambda_grl=1.0):
        """
        前向传播主函数

        Args:
            mtf_features: Tensor, 形状 [Batch, Matrix_Size, Matrix_Size] 或 [Batch, C, M, M]
                          来自 TFMPTF 的矩阵特征
            lambda_grl: float, 梯度反转系数 (0 到 1)
                        0: 不进行对抗 (预训练阶段)
                        1: 完全对抗 (域适应阶段)

        Returns:
            class_logits: Tensor,形状 [Batch, Num_Classes]
                          分类器的输出
            domain_pred: Tensor, 形状 [Batch, 1]
                         判别器的输出
        """
        # 确保输入是 4D 的 [B, C, M, M]
        if mtf_features.dim() == 3:
            mtf_features = mtf_features.unsqueeze(1)

        # 1. 特征提取
        x = self.feature_head(mtf_features)

        # 2. 分类分支
        class_logits = self.classifier(x)

        # 3. 域判别分支 (带梯度反转)
        # 使用自定义 Function 实现 GRL
        x_rev = GradientReversalFunction.apply(x, lambda_grl)
        domain_pred = self.discriminator(x_rev)

        return class_logits, domain_pred