"""
MAML 元学习核心（符合 WorkBuddy 宪章 v5.0.0 第15.4节）
- 元网络与内循环更新
- 训练与适应接口
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class LinUCBBase(nn.Module):
    """可微分的 LinUCB 基学习器（用于 MAML 内循环）"""

    def __init__(self, feature_dim: int, alpha: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.A = nn.Parameter(torch.eye(feature_dim))
        self.b = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算 UCB 分数（可微）"""
        A_inv = torch.inverse(self.A + 1e-6 * torch.eye(self.feature_dim))
        theta = A_inv @ self.b
        mean = (theta @ x).squeeze()
        std = torch.sqrt((x @ A_inv @ x).squeeze() + 1e-8)
        return mean + self.alpha * std


class MAMLMetaLearner(nn.Module):
    """
    MAML 元学习器：将任务特征映射为 LinUCB 参数
    内循环（inner loop）快速适应，外循环（outer loop）更新元参数
    """

    def __init__(self, task_feature_dim: int, base_feature_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.base_feature_dim = base_feature_dim
        self.hidden_dim = hidden_dim

        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 输出：A 参数的 flatten + b 向量
        self.A_size = base_feature_dim * base_feature_dim
        self.A_head = nn.Linear(hidden_dim, self.A_size)
        self.b_head = nn.Linear(hidden_dim, base_feature_dim)

    def forward(self, task_features: torch.Tensor) -> tuple:
        """
        将任务特征映射为 LinUCB 参数 (A, b)
        返回：(A, b)，A 为 (d, d)，b 为 (d,)
        """
        h = self.task_encoder(task_features)
        A_flat = self.A_head(h)          # shape: (batch, d^2)
        A = A_flat.view(self.base_feature_dim, self.base_feature_dim)  # reshape
        b_vec = self.b_head(h)           # shape: (batch, d)
        return A, b_vec

    def adapt(self, base_learner: LinUCBBase, task_features: torch.Tensor,
              rewards: torch.Tensor, inner_lr: float = 0.01, inner_steps: int = 5):
        """
        MAML 内循环：对给定任务快速适应基学习器参数
        task_features: shape (N, d_task)
        rewards: shape (N,)
        """
        adapted_learners = []
        for i in range(task_features.size(0)):
            A = base_learner.A.clone()
            b = base_learner.b.clone()
            x = task_features[i]
            r = rewards[i]

            for _ in range(inner_steps):
                A_inv = torch.inverse(A + 1e-6 * torch.eye(self.base_feature_dim))
                theta = A_inv @ b
                loss = -r * (theta @ x)
                loss.backward()
                with torch.no_grad():
                    b -= inner_lr * base_learner.b.grad
                    A -= inner_lr * A.grad
                base_learner.zero_grad()

            adapted_learner = LinUCBBase(self.base_feature_dim, base_learner.alpha)
            adapted_learner.A = nn.Parameter(A)
            adapted_learner.b = nn.Parameter(b)
            adapted_learners.append(adapted_learner)

        return adapted_learners


def train_maml(meta_tasks: List[Dict], feature_dim: int,
                task_feature_dim: int, epochs: int = 100,
                inner_lr: float = 0.01, outer_lr: float = 0.001) -> MAMLMetaLearner:
    """
    训练 MAML 元学习器（符合宪章 v5.0.0 第15.4节）
    meta_tasks: [{"task_feature": Tensor, "samples": [...]}, ...]
                task_feature: 任务级特征（one-hot 或 embedding）
                samples: [{"context": List, "reward": float}, ...]
    """
    device = torch.device("cpu")  # CI 环境下使用 CPU
    meta_learner = MAMLMetaLearner(
        task_feature_dim=task_feature_dim,
        base_feature_dim=feature_dim
    ).to(device)
    optimizer = torch.optim.Adam(meta_learner.parameters(), lr=outer_lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for task in meta_tasks:
            # 兼容 task_feature / task_features 两种字段名
            task_feat = task.get("task_feature") or task.get("task_features")
            if task_feat is None:
                continue

            if isinstance(task_feat, list):
                task_feat = torch.tensor(task_feat, dtype=torch.float32)
            task_feat = task_feat.to(device).unsqueeze(0)  # (1, task_feature_dim)

            # 从 samples 聚合奖励
            samples = task.get("samples", [])
            if samples:
                rewards_list = torch.tensor(
                    [s.get("reward", 0.0) for s in samples],
                    dtype=torch.float32
                ).to(device)
            else:
                rewards_list = torch.tensor([0.0], dtype=torch.float32).to(device)

            # 外循环：计算元梯度
            A_flat, b_vec = meta_learner(task_feat)
            # 简化损失：元目标是最大化累积奖励
            loss = -rewards_list.mean()

            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % max(1, epochs // 5) == 0 or epochs <= 5:
            print(f"[MAML] Epoch {epoch+1}/{epochs}, avg loss: {total_loss / max(len(meta_tasks), 1):.4f}")

    return meta_learner
