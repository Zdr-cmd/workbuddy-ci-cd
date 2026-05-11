"""
P0 MAML 元学习训练测试（轻量级，CPU 可运行）
符合宪章 v5.0.0 第15.4节要求
"""

import pytest
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.meta_learner import MAMLMetaLearner, LinUCBBase, train_maml


class TestLinUCBBase:
    """测试可微分 LinUCB 基学习器"""

    def test_forward_pass(self):
        """前向传播应返回 UCB 分数"""
        model = LinUCBBase(feature_dim=12, alpha=1.0)
        x = torch.randn(12)
        ucb = model(x)
        assert isinstance(ucb.item(), float)
        assert not torch.isnan(ucb)

    def test_parameters_init(self):
        """参数初始化正确"""
        model = LinUCBBase(feature_dim=8)
        assert model.feature_dim == 8
        assert model.alpha == 1.0
        assert model.A.shape == (8, 8)
        assert model.b.shape == (8,)


class TestMAMLMetaLearner:
    """测试 MAML 元学习器"""

    def test_forward_returns_correct_shapes(self):
        """前向传播返回正确维度的 (A, b)"""
        meta = MAMLMetaLearner(task_feature_dim=4, base_feature_dim=12)
        task_feat = torch.randn(4)
        A, b = meta(task_feat)
        assert A.shape == (12, 12)
        assert b.shape == (12,)

    def test_forward_single_sample(self):
        """单个样本前向传播"""
        meta = MAMLMetaLearner(task_feature_dim=3, base_feature_dim=8)
        task_feat = torch.randn(3)
        A, b = meta(task_feat)
        assert A.shape == (8, 8)
        assert b.shape == (8,)


class TestTrainMAML:
    """测试 MAML 训练循环"""

    @pytest.fixture
    def dummy_meta_tasks(self):
        """生成3个模拟任务"""
        tasks = []
        for task_id in range(3):
            samples = []
            for _ in range(10):
                context = np.random.randn(12).tolist()
                reward = float(np.random.rand() * 0.5 + 0.5)
                samples.append({"context": context, "reward": reward})
            tasks.append({
                "task_feature": [1.0 if i == task_id else 0.0 for i in range(3)],
                "samples": samples
            })
        return tasks

    def test_train_maml_forward_only(self, dummy_meta_tasks):
        """仅测试前向传播（不训练，验证数据流正确）"""
        meta = MAMLMetaLearner(task_feature_dim=3, base_feature_dim=12)
        for task in dummy_meta_tasks:
            task_feat = torch.tensor(task["task_feature"], dtype=torch.float32)
            A, b = meta(task_feat)
            assert A.shape == (12, 12)
            assert b.shape == (12,)

    def test_train_maml_training_loop_runs(self, dummy_meta_tasks):
        """训练循环能够运行（无梯度爆炸/崩溃）"""
        try:
            meta = train_maml(
                dummy_meta_tasks,
                feature_dim=12,
                task_feature_dim=3,
                epochs=2  # 仅2个epoch保证CI快速完成
            )
            assert meta is not None
            assert isinstance(meta, MAMLMetaLearner)
        except Exception as e:
            pytest.fail(f"训练循环失败: {e}")

    def test_train_maml_empty_tasks(self):
        """空任务列表应不崩溃"""
        result = train_maml([], feature_dim=12, task_feature_dim=3, epochs=1)
        assert isinstance(result, MAMLMetaLearner)

    def test_train_maml_missing_samples(self):
        """缺少 samples 的任务应安全处理"""
        tasks = [{"task_feature": [1.0, 0.0, 0.0]}]
        result = train_maml(tasks, feature_dim=12, task_feature_dim=3, epochs=1)
        assert isinstance(result, MAMLMetaLearner)

    def test_train_maml_tensor_task_feature(self):
        """直接传 Tensor 也应工作"""
        tasks = [{"task_features": torch.randn(4), "samples": []}]
        result = train_maml(tasks, feature_dim=8, task_feature_dim=4, epochs=1)
        assert isinstance(result, MAMLMetaLearner)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
