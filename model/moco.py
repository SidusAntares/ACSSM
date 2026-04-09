import torch
from torch import nn
import torch.nn.functional as F


class AdaMoCo(nn.Module):
    """
    单流MoCo对比学习模块（适配融合特征）
    核心思想：对比学习作用于门控融合后的特征

    ⚠️ 重要：此模块只负责对比学习，不包含任何分类器！
    """

    def __init__(self, feature_dim=128, num_classes=7, queue_size=4096, momentum=0.999, temperature=0.07):
        """
        Args:
            feature_dim: 特征维度（如128）
            num_classes: 类别数（用于队列标签管理，不影响对比学习）
            queue_size: 队列大小（如4096）
            momentum: 队列更新动量（如0.999）
            temperature: 温度参数（如0.07）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # ✅ 单流队列（融合特征）
        # 队列存储负样本特征 [feature_dim, queue_size]
        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
        # 队列标签用于分析和调试 [queue_size]
        self.register_buffer("queue_labels", torch.zeros(queue_size, dtype=torch.long))
        # 队列指针 [1]
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 初始化队列（归一化）
        self.register_buffer("queue",
            F.normalize(torch.randn(feature_dim, queue_size), dim=0)
        )

        # ✅ 确保队列大小是batch_size的整数倍（可选）
        print(f"✅ AdaMoCo初始化: feature_dim={feature_dim}, queue_size={queue_size}, T={temperature}")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        """
        原子操作：更新队列（先进先出）

        Args:
            keys: 键特征 [B, feature_dim]
            labels: 键标签 [B]
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 处理边界情况（队列循环）
        if ptr + batch_size > self.queue_size:
            # 第一部分：填充到队列末尾
            first_part = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first_part].T
            self.queue_labels[ptr:] = labels[:first_part]

            # 第二部分：从队列开头继续填充
            second_part = batch_size - first_part
            self.queue[:, :second_part] = keys[first_part:].T
            self.queue_labels[:second_part] = labels[first_part:]
        else:
            # 直接填充
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_labels[ptr:ptr + batch_size] = labels

        # ✅ 更新指针（循环）
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, fused_q, fused_k, labels):
        """
        单流对比学习前向传播

        ⚠️ 核心：只计算对比损失，不涉及分类！

        Args:
            fused_q: 查询样本的融合特征 [B, feature_dim]
            fused_k: 键样本的融合特征 [B, feature_dim]
            labels: 样本标签 [B]（仅用于队列管理，不影响损失计算）

        Returns:
            loss: InfoNCE对比损失 [1]
            logits: 对比logits [B, 1+K]（可用于分析）
        """
        batch_size = fused_q.shape[0]

        # ✅ 归一化特征（L2归一化）
        fused_q = F.normalize(fused_q, dim=1)  # [B, feature_dim]
        fused_k = F.normalize(fused_k, dim=1)  # [B, feature_dim]

        # ✅ 正样本：查询和键的相似度 [B, 1]
        # 使用点积计算相似度
        l_pos = torch.einsum('nc,nc->n', [fused_q, fused_k]).unsqueeze(-1)  # [B, 1]

        # ✅ 负样本：查询和队列的相似度 [B, K]
        # 队列中的特征作为负样本
        l_neg = torch.einsum('nc,ck->nk', [fused_q, self.queue.clone().detach()])  # [B, queue_size]

        # ✅ 合并正负样本 [B, 1+K]
        # 第0列是正样本，后面是负样本
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+queue_size]

        # ✅ 应用温度参数（控制softmax的尖锐程度）
        logits /= self.temperature

        # ✅ InfoNCE损失
        # 标签都是0，因为正样本在第0列
        labels_infonce = torch.zeros(batch_size, dtype=torch.long, device=fused_q.device)
        loss = F.cross_entropy(logits, labels_infonce)

        # ✅ 更新队列（使用键特征）
        # 注意：队列更新是原地操作，不参与梯度计算
        self._dequeue_and_enqueue(fused_k, labels)

        return loss

    @torch.no_grad()
    def get_queue_stats(self):
        """
        获取队列统计信息（用于调试和分析）

        Returns:
            dict: 队列状态统计
        """
        queue_norm = torch.norm(self.queue, dim=0)
        return {
            'queue_size': self.queue_size,
            'current_ptr': int(self.queue_ptr),
            'queue_norm_mean': queue_norm.mean().item(),
            'queue_norm_std': queue_norm.std().item(),
            'unique_labels': len(torch.unique(self.queue_labels[self.queue_labels > 0])),
            'label_distribution': torch.bincount(self.queue_labels, minlength=self.num_classes).cpu().numpy()
        }

    def reset_queue(self):
        """
        重置队列（用于新实验）
        """
        self.queue = F.normalize(torch.randn(self.feature_dim, self.queue_size), dim=0).to(self.queue.device)
        self.queue_labels.zero_()
        self.queue_ptr.zero_()
        print("✅ 队列已重置")

    def get_queue(self):
        """
        获取当前队列状态（只读）

        Returns:
            dict: 队列和标签
        """
        return {
            'queue': self.queue.detach().clone(),
            'queue_labels': self.queue_labels.detach().clone()
        }

    def compute_similarity_matrix(self, fused_q):
        """
        计算查询特征与队列的相似度矩阵（用于可视化和分析）

        Args:
            fused_q: 查询特征 [B, feature_dim]

        Returns:
            sim_matrix: 相似度矩阵 [B, queue_size]
        """
        fused_q = F.normalize(fused_q, dim=1)
        sim_matrix = torch.einsum('nc,ck->nk', [fused_q, self.queue])
        return sim_matrix

    def __repr__(self):
        return (f"AdaMoCo(feature_dim={self.feature_dim}, "
                f"queue_size={self.queue_size}, "
                f"temperature={self.temperature}, "
                f"momentum={self.momentum})")