"""
P0 因果推断测试
符合宪章 v5.0.0 第15.3节要求
"""

import pytest
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.causal_inference import CausalImpactEvaluator


class TestDiDEvaluation:
    """测试双重差分因果效应评估"""

    def test_evaluate_evolution_insufficient_data(self):
        """空 DataFrame 应返回 Insufficient data"""
        evaluator = CausalImpactEvaluator()
        evaluator.df = pd.DataFrame()
        result = evaluator.evaluate_evolution(
            "model_selection_tweak",
            pd.Timestamp("2026-05-01")
        )
        assert result.get("error") == "Insufficient data"

    def test_evaluate_evolution_valid_data(self):
        """有效数据应返回因果效应评估"""
        dates = pd.date_range("2026-04-01", periods=100, freq="D")
        np.random.seed(42)
        df = pd.DataFrame({
            "timestamp": dates,
            "user_id": [f"u{i//2}" for i in range(100)],
            "user_group": ["experiment" if i < 50 else "control" for i in range(100)],
            "success": [1.0 if i < 50 else 0.8 + np.random.rand() * 0.1 for i in range(100)]
        })
        evaluator = CausalImpactEvaluator()
        evaluator.df = df
        result = evaluator.evaluate_evolution(
            "strategy_switch",
            pd.Timestamp("2026-04-20"),
            metric="success"
        )
        assert "causal_effect" in result or "error" in result
        assert "p_value" in result

    def test_evaluate_evolution_unknown_metric(self):
        """未知 metric 应不崩溃"""
        evaluator = CausalImpactEvaluator()
        evaluator.df = pd.DataFrame({
            "timestamp": pd.date_range("2026-04-01", periods=10),
            "user_id": ["u0"] * 10,
            "user_group": ["experiment"] * 5 + ["control"] * 5,
            "unknown_metric": [1.0] * 10
        })
        result = evaluator.evaluate_evolution(
            "param_adjust",
            pd.Timestamp("2026-04-05"),
            metric="unknown_metric"
        )
        assert isinstance(result, dict)


class TestPSMMatching:
    """测试倾向性得分匹配（sklearn 实现）"""

    def test_psm_empty_dataframe(self):
        """空 DataFrame 返回空列表"""
        evaluator = CausalImpactEvaluator()
        evaluator.df = pd.DataFrame()
        treated, control = evaluator.propensity_score_matching()
        assert treated == []
        assert control == []

    def test_psm_balanced_data(self):
        """平衡数据应能完成匹配"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "user_id": [f"u{i}" for i in range(n)],
            "user_group": ["experiment" if i < 50 else "control" for i in range(n)],
            "cost_usd": np.random.uniform(0.01, 0.05, n),
            "latency": np.random.uniform(0.5, 2.0, n)
        })
        evaluator = CausalImpactEvaluator()
        evaluator.df = df
        treated, control = evaluator.propensity_score_matching(
            covariates=["cost_usd", "latency"],
            caliper=0.05
        )
        assert isinstance(treated, list)
        assert isinstance(control, list)
        # 匹配数应相等
        assert len(treated) == len(control)

    def test_psm_one_to_one(self):
        """1:1 匹配：处理组和对照组数量应相等"""
        np.random.seed(99)
        df = pd.DataFrame({
            "user_id": [f"u{i}" for i in range(60)],
            "user_group": ["experiment" if i < 30 else "control" for i in range(60)],
            "cost_usd": [0.02] * 60,
            "latency": [1.0] * 60
        })
        evaluator = CausalImpactEvaluator()
        evaluator.df = df
        treated, control = evaluator.propensity_score_matching(
            covariates=["cost_usd", "latency"],
            caliper=0.1
        )
        # 倾向得分完全相同时，应1:1匹配
        assert len(treated) == len(control)

    def test_psm_custom_covariates(self):
        """自定义协变量应生效"""
        np.random.seed(7)
        n = 40
        df = pd.DataFrame({
            "user_id": [f"u{i}" for i in range(n)],
            "user_group": ["experiment" if i < 20 else "control" for i in range(n)],
            "cost_usd": np.random.rand(n) * 0.04,
            "latency": np.random.rand(n) * 1.5 + 0.5,
            "extra_col": np.random.rand(n)  # 不在协变量中，应被忽略
        })
        evaluator = CausalImpactEvaluator()
        evaluator.df = df
        treated, control = evaluator.propensity_score_matching(
            covariates=["cost_usd", "latency"],  # extra_col 不传入
            caliper=0.05
        )
        assert len(treated) == len(control)


class TestSummary:
    """测试评估摘要"""

    def test_summary_no_data(self):
        evaluator = CausalImpactEvaluator()
        evaluator.df = pd.DataFrame()
        summary = evaluator.summary()
        assert summary["status"] == "no_data"
        assert summary["total_records"] == 0

    def test_summary_with_data(self):
        df = pd.DataFrame({
            "user_id": ["u0", "u1", "u2"],
            "timestamp": pd.date_range("2026-04-01", periods=3),
            "success": [True, False, True],
            "cost_usd": [0.01, 0.02, 0.01]
        })
        evaluator = CausalImpactEvaluator()
        evaluator.df = df
        summary = evaluator.summary()
        assert summary["status"] == "ready"
        assert summary["total_records"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
