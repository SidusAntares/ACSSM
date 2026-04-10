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


import torch
from torch import nn
import torch.nn.functional as F


class AdaMoCo3D(nn.Module):
    """
    三维度AdaMoCo对比学习模块（保留组维度信息）

    核心特性：
    - 输入：[B, G, C] - 批次×组×特征维度
    - 队列：[queue_size, G, C] - 队列大小×组×特征维度
    - 支持多种对比策略：逐组对比、全局对比
    - 动态适应输入维度

    ⚠️ 重要：此模块只负责对比学习，不包含任何分类器！
    """

    def __init__(
            self,
            feature_dim=128,
            num_groups=16,
            num_classes=7,
            queue_size=4096,
            momentum=0.999,
            temperature=0.07,
            contrast_mode='group_wise',  # 'group_wise', 'global'
            dynamic_resize=True  # 是否启用动态维度调整
    ):
        """
        Args:
            feature_dim: 特征维度（如128）
            num_groups: 组数量（如16）
            num_classes: 类别数（用于队列标签管理）
            queue_size: 队列大小（如4096）
            momentum: 队列更新动量（如0.999）
            temperature: 温度参数（如0.07）
            contrast_mode: 对比模式
                - 'group_wise': 逐组对比（每个组独立计算对比损失）
                - 'global': 全局对比（压缩组维度后对比）
            dynamic_resize: 是否动态调整队列维度以匹配输入
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.dynamic_resize = dynamic_resize

        # 初始化队列（使用初始维度）
        self._init_queue(feature_dim, num_groups, queue_size)

        print(f"✅ AdaMoCo3D初始化: feature_dim={feature_dim}, num_groups={num_groups}, "
              f"queue_size={queue_size}, T={temperature}, mode={contrast_mode}, "
              f"dynamic_resize={dynamic_resize}")

    def _init_queue(self, feature_dim, num_groups, queue_size):
        """初始化队列"""
        # 队列：[queue_size, num_groups, feature_dim]
        self.register_buffer("queue", torch.randn(queue_size, num_groups, feature_dim))
        self.queue = F.normalize(self.queue, dim=2)
        self.register_buffer("queue_labels", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 保存当前队列维度
        self.current_feature_dim = feature_dim
        self.current_num_groups = num_groups
        self.current_queue_size = queue_size

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        """
        更新队列：移除最旧的批次，添加新批次
        keys: [B, G, C] - 键特征
        labels: [B] 或 [B, G] - 标签
        """
        batch_size = keys.shape[0]

        # ✅ 处理标签维度
        if labels.dim() == 2:  # [B, G]
            if labels.shape[1] == keys.shape[1]:  # 组数匹配
                labels = labels[:, 0]  # 取第一组的标签
            else:
                raise ValueError(f"标签组数 {labels.shape[1]} 与特征组数 {keys.shape[1]} 不匹配")
        elif labels.dim() == 1:  # [B] - 正常情况
            pass
        else:
            raise ValueError(f"不支持的标签维度: {labels.dim()}")

        ptr = int(self.queue_ptr)

        # ✅ 动态调整队列大小以适应batch_size
        if self.queue_size % batch_size != 0:
            # 计算新的队列大小（最接近的整数倍）
            new_queue_size = ((self.queue_size // batch_size) + 1) * batch_size
            # print(f"⚠️ 调整队列大小: {self.queue_size} -> {new_queue_size} (适应batch_size={batch_size})")
            self._resize_queue(new_queue_size, self.current_num_groups, self.current_feature_dim)

        # 替换队列中的样本
        self.queue[ptr:ptr + batch_size] = keys
        self.queue_labels[ptr:ptr + batch_size] = labels

        # 移动指针
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def _resize_queue(self, new_queue_size, new_num_groups, new_feature_dim):
        """调整队列大小和维度"""
        device = self.queue.device
        dtype = self.queue.dtype

        # 保存旧队列的部分数据（如果新队列更小）
        if new_queue_size < self.queue_size:
            old_queue = self.queue[:new_queue_size]
            old_labels = self.queue_labels[:new_queue_size]
        else:
            old_queue = self.queue
            old_labels = self.queue_labels

        # 创建新队列
        new_queue = torch.randn(new_queue_size, new_num_groups, new_feature_dim, device=device, dtype=dtype)
        new_queue = F.normalize(new_queue, dim=2)

        # 复制旧数据（如果维度匹配）
        if new_num_groups == self.current_num_groups and new_feature_dim == self.current_feature_dim:
            min_size = min(new_queue_size, self.queue_size)
            new_queue[:min_size] = old_queue[:min_size]
            new_labels = torch.zeros(new_queue_size, dtype=torch.long, device=device)
            new_labels[:min_size] = old_labels[:min_size]
        else:
            new_labels = torch.zeros(new_queue_size, dtype=torch.long, device=device)

        # 更新缓冲区
        self.register_buffer("queue", new_queue)
        self.register_buffer("queue_labels", new_labels)
        self.queue_ptr[0] = 0  # 重置指针

        # 更新状态
        self.current_queue_size = new_queue_size
        self.current_num_groups = new_num_groups
        self.current_feature_dim = new_feature_dim
        self.queue_size = new_queue_size

        # print(f"✅ 队列已调整: size={new_queue_size}, groups={new_num_groups}, dim={new_feature_dim}")

    def _compute_group_wise_contrast(self, fused_q, fused_k):
        """
        逐组对比学习：每个组独立计算对比损失
        """
        B, G, C = fused_q.shape

        losses = []
        logits_list = []

        for g in range(G):
            # 提取第g组特征
            q_g = fused_q[:, g, :]  # [B, C]
            k_g = fused_k[:, g, :]  # [B, C]

            # 归一化
            q_g = F.normalize(q_g, dim=1)
            k_g = F.normalize(k_g, dim=1)

            # 正样本
            l_pos = torch.einsum('nc,nc->n', [q_g, k_g]).unsqueeze(-1)  # [B, 1]

            # 从队列中提取对应组 [K, C]
            queue_g = self.queue[:, g, :].clone().detach()  # [K, C]
            queue_g = queue_g.transpose(0, 1)  # [C, K]
            l_neg = torch.einsum('nc,ck->nk', [q_g, queue_g])  # [B, K]

            # 合并
            logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature  # [B, 1+K]
            logits_list.append(logits)

            # 计算损失
            labels = torch.zeros(B, dtype=torch.long, device=fused_q.device)
            loss_g = F.cross_entropy(logits, labels)
            losses.append(loss_g)

        # 平均所有组的损失
        total_loss = torch.stack(losses).mean()
        logits_tensor = torch.stack(logits_list)  # [G, B, 1+K]

        return total_loss, logits_tensor

    def _compute_global_contrast(self, fused_q, fused_k):
        """
        全局对比学习：压缩组维度后进行对比

        Args:
            fused_q: [B, G, C]
            fused_k: [B, G, C]

        Returns:
            loss: 对比损失
            logits: logits [B, 1+K]
        """
        # 压缩组维度（平均池化）
        fused_q_global = fused_q.mean(dim=1)  # [B, C]
        fused_k_global = fused_k.mean(dim=1)  # [B, C]

        # 归一化
        fused_q_global = F.normalize(fused_q_global, dim=1)
        fused_k_global = F.normalize(fused_k_global, dim=1)

        # 正样本
        l_pos = torch.einsum('nc,nc->n', [fused_q_global, fused_k_global]).unsqueeze(-1)  # [B, 1]

        # 负样本（压缩队列）
        queue_global = self.queue.mean(dim=1).clone().detach()  # [K, C]
        queue_global = queue_global.t()  # [C, K]
        l_neg = torch.einsum('nc,ck->nk', [fused_q_global, queue_global])  # [B, K]

        # 合并
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature  # [B, 1+K]

        # 计算损失
        labels = torch.zeros(fused_q.shape[0], dtype=torch.long, device=fused_q.device)
        loss = F.cross_entropy(logits, labels)

        return loss, logits

    def forward(self, fused_q, fused_k, labels):
        """
        前向传播

        Args:
            fused_q: [B, G, C] - 查询特征
            fused_k: [B, G, C] - 键特征
            labels: [B] 或 [B, G] - 标签

        Returns:
            loss: 对比损失
            logits: logits
        """
        B, G, C = fused_q.shape

        # ✅ 检查并动态调整队列维度
        if self.dynamic_resize:
            need_resize = False
            resize_info = []

            # 检查组数量
            if G != self.current_num_groups:
                resize_info.append(f"组数 {self.current_num_groups}->{G}")
                need_resize = True

            # 检查特征维度
            if C != self.current_feature_dim:
                resize_info.append(f"特征维度 {self.current_feature_dim}->{C}")
                need_resize = True

            # 重新初始化队列
            if need_resize:
                # print(f"🔄 重新初始化队列: {' | '.join(resize_info)}")
                self._resize_queue(self.queue_size, G, C)

        # 归一化输入
        fused_q = F.normalize(fused_q, dim=2)
        fused_k = F.normalize(fused_k, dim=2)

        # 入队出队
        self._dequeue_and_enqueue(fused_k, labels)

        # 计算对比损失
        if self.contrast_mode == 'group_wise':
            loss, logits = self._compute_group_wise_contrast(fused_q, fused_k)
        elif self.contrast_mode == 'global':
            loss, logits = self._compute_global_contrast(fused_q, fused_k)
        else:
            raise ValueError(f"Unknown contrast mode: {self.contrast_mode}")

        return loss, logits

    @torch.no_grad()
    def get_queue_stats(self):
        """
        获取队列统计信息（用于调试和分析）

        Returns:
            dict: 队列状态统计
        """
        queue_norm = torch.norm(self.queue, dim=2)  # [K, G]
        return {
            'queue_size': self.queue_size,
            'num_groups': self.current_num_groups,
            'feature_dim': self.current_feature_dim,
            'current_ptr': int(self.queue_ptr),
            'queue_norm_mean': queue_norm.mean().item(),
            'queue_norm_std': queue_norm.std().item(),
            'queue_norm_per_group': queue_norm.mean(dim=0).cpu().numpy(),  # 每组的平均范数
            'unique_labels': len(torch.unique(self.queue_labels[self.queue_labels > 0])),
            'label_distribution': torch.bincount(self.queue_labels, minlength=self.num_classes).cpu().numpy()
        }

    def reset_queue(self):
        """
        重置队列（用于新实验）
        """
        self._init_queue(self.current_feature_dim, self.current_num_groups, self.queue_size)
        # print("✅ 3D队列已重置")

    def get_queue(self):
        """
        获取当前队列状态（只读）

        Returns:
            dict: 队列和标签
        """
        return {
            'queue': self.queue.detach().clone(),  # [K, G, C]
            'queue_labels': self.queue_labels.detach().clone()
        }

    def compute_similarity_matrix(self, fused_q, group_idx=None):
        """
        计算查询特征与队列的相似度矩阵

        Args:
            fused_q: 查询特征 [B, G, C] 或 [B, C]
            group_idx: 指定组索引（None表示全局）

        Returns:
            sim_matrix: 相似度矩阵
                - 如果group_idx is None: [B, K]
                - 如果group_idx is not None: [B, K]
        """
        # 处理输入
        if fused_q.dim() == 2:
            fused_q = fused_q.unsqueeze(1).expand(-1, self.current_num_groups, -1)

        fused_q = F.normalize(fused_q, dim=2)  # [B, G, C]

        if group_idx is None:
            # 全局相似度
            fused_q_global = fused_q.mean(dim=1)  # [B, C]
            queue_global = self.queue.mean(dim=1).clone()  # [K, C]
            sim_matrix = torch.einsum('nc,kc->nk', [fused_q_global, queue_global])
        else:
            # 特定组的相似度
            fused_q_g = fused_q[:, group_idx, :]  # [B, C]
            queue_g = self.queue[:, group_idx, :].clone()  # [K, C]
            sim_matrix = torch.einsum('nc,kc->nk', [fused_q_g, queue_g])

        return sim_matrix

    def compute_inter_group_similarity(self, fused_q):
        """
        计算组间相似度（分析组特征的多样性）

        Args:
            fused_q: [B, G, C]

        Returns:
            sim_matrix: 组间相似度矩阵 [G, G]
        """
        if fused_q.dim() == 2:
            return torch.eye(self.current_num_groups, device=fused_q.device)

        # 平均批次维度
        group_features = fused_q.mean(dim=0)  # [G, C]
        group_features = F.normalize(group_features, dim=1)

        # 计算组间相似度
        sim_matrix = torch.mm(group_features, group_features.T)  # [G, G]

        return sim_matrix

    def __repr__(self):
        return (f"AdaMoCo3D(feature_dim={self.current_feature_dim}, "
                f"num_groups={self.current_num_groups}, "
                f"queue_size={self.queue_size}, "
                f"temperature={self.temperature}, "
                f"mode={self.contrast_mode})")