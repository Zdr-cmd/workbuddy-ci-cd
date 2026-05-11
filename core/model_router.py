"""
动态模型路由器（符合 WorkBuddy 章程 v4.5.0）
- 基于成功率和成本的加权路由
- 滑动窗口统计（窗口大小可配置）
- 自动持久化（定期保存 + 启动加载）
- 用户偏好设置（cost/latency/balanced）
"""

import json
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional


class ModelPreference:
    """用户偏好管理（成本优先/延迟优先/平衡）"""

    def __init__(self, config_file: str = "state/model_pref.json"):
        self.config_file = config_file
        self.mode = "balanced"           # cost / latency / balanced
        self.custom_weights = None
        self.load()

    def load(self):
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.mode = data.get("mode", "balanced")
                    self.custom_weights = data.get("custom_weights")
            except Exception:
                pass

    def save(self):
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump({
                "mode": self.mode,
                "custom_weights": self.custom_weights
            }, f, indent=2, ensure_ascii=False)

    def set_mode(self, mode: str):
        if mode not in ["cost", "latency", "balanced"]:
            return False, "模式必须是 cost/latency/balanced"
        self.mode = mode
        self.save()
        return True, f"路由模式已切换为 {mode}"

    def get_weights(self) -> Dict[str, float]:
        """返回 (success, cost, latency) 权重"""
        if self.custom_weights:
            return self.custom_weights
        if self.mode == "cost":
            return {"success": 0.4, "cost": 0.5, "latency": 0.1}
        elif self.mode == "latency":
            return {"success": 0.4, "cost": 0.1, "latency": 0.5}
        else:  # balanced
            return {"success": 0.5, "cost": 0.25, "latency": 0.25}


class WeightedModelRouter:
    """
    基于加权评分的模型路由器
    - 记录每个 (model, task_type) 的成功率和平均成本（滑动窗口）
    - 根据用户偏好选择最优模型
    """

    def __init__(self,
                 config: dict = None,
                 stats_file: str = "state/model_stats.json",
                 auto_save_interval: int = 60,
                 window_size: int = 50):
        self.stats_file = stats_file
        self.auto_save_interval = auto_save_interval
        self.window_size = window_size
        self.stats: Dict[str, Dict[str, deque]] = {}  # key -> {"success": deque, "cost": deque, "latency": deque}
        self.preference = ModelPreference()
        
        # ✅ 从配置读取模型列表和任务映射
        self.config = config if config else {}
        self._load_task_mapping()
        
        self._load_stats()
        self._start_auto_save()

    def _load_task_mapping(self):
        """从配置加载任务类型到模型的映射"""
        routing_config = self.config.get("routing", {})
        self.task_mapping = routing_config.get("task_type_mapping", {})
        
        # 提取所有启用的模型
        models_config = self.config.get("models", {})
        self.enabled_models = [m["name"] for m in models_config.get("enabled", [])]
        
        # 如果没有配置，使用默认值
        if not self.task_mapping:
            self.task_mapping = {
                "code": ["qwen3:14b", "hy3-preview"],
                "default": ["hy3-preview", "qwen3:14b"]
            }
        if not self.enabled_models:
            self.enabled_models = ["hy3-preview", "qwen3:14b", "gemma3:12b"]

    # ---------- 持久化 ----------
    def _load_stats(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, val in data.items():
                    self.stats[key] = {
                        "success": deque(val.get("success", []), maxlen=self.window_size),
                        "cost": deque(val.get("cost", []), maxlen=self.window_size),
                        "latency": deque(val.get("latency", []), maxlen=self.window_size)
                    }
            except Exception as e:
                print(f"⚠️ 加载路由统计失败: {e}")

    def _save_stats(self):
        data = {}
        for key, stat in self.stats.items():
            data[key] = {
                "success": list(stat["success"]),
                "cost": list(stat["cost"]),
                "latency": list(stat.get("latency", []))
            }
        try:
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存路由统计失败: {e}")

    def _start_auto_save(self):
        def save_loop():
            while True:
                time.sleep(self.auto_save_interval)
                self._save_stats()
        thread = threading.Thread(target=save_loop, daemon=True)
        thread.start()

    # ---------- 核心路由 ----------
    def select_model(self, task_type: str, context: dict = None) -> str:
        """根据当前统计和用户偏好选择最佳模型"""
        candidates = self._get_candidates(task_type)
        weights = self.preference.get_weights()
        best_model = None
        best_score = -1.0

        for model in candidates:
            stat_key = f"{model}_{task_type}"
            if stat_key not in self.stats or len(self.stats[stat_key]["success"]) == 0:
                score = 0.5   # 无历史数据，中性分数
            else:
                stats = self.stats[stat_key]
                success_rate = sum(stats["success"]) / len(stats["success"])
                avg_cost = sum(stats["cost"]) / len(stats["cost"]) if stats["cost"] else 0.0

                # 成本归一化（假设最大可接受成本 0.05 USD）
                cost_score = 1.0 - min(avg_cost / 0.05, 1.0)
                
                # ✅ 真正的延迟分数（不再用成本占位）
                latency_list = stats.get("latency", [])
                if latency_list:
                    avg_latency = sum(latency_list) / len(latency_list)
                    # 延迟归一化（假设最大可接受延迟 5.0 秒）
                    latency_score = 1.0 - min(avg_latency / 5.0, 1.0)
                else:
                    latency_score = 0.5  # 无延迟数据时取中性分数

                score = (weights["success"] * success_rate +
                         weights["cost"] * cost_score +
                         weights["latency"] * latency_score)

            if score > best_score:
                best_score = score
                best_model = model

        return best_model if best_model else self._static_default(task_type)

    def update_stats(self, model: str, task_type: str, success: bool, cost: float, latency: float = 0.0):
        """更新滑动窗口统计（每次请求后调用）"""
        key = f"{model}_{task_type}"
        if key not in self.stats:
            self.stats[key] = {
                "success": deque(maxlen=self.window_size),
                "cost": deque(maxlen=self.window_size),
                "latency": deque(maxlen=self.window_size)
            }
        self.stats[key]["success"].append(1 if success else 0)
        self.stats[key]["cost"].append(cost)
        self.stats[key]["latency"].append(latency)
        # 注意：不立即保存，由自动保存线程负责

    # ---------- 辅助方法 ----------
    def _get_candidates(self, task_type: str) -> List[str]:
        """根据配置或默认值获取候选模型列表"""
        return self.task_mapping.get(task_type, self.task_mapping.get("default", ["hy3-preview"]))

    def _static_default(self, task_type: str) -> str:
        static = {
            "code_generation": "qwen3:14b",
            "long_text_analysis": "hy3-preview",
            "default": "hy3-preview"
        }
        return static.get(task_type, "hy3-preview")

    # ---------- 用户偏好覆盖 ----------
    def set_preference_mode(self, mode: str):
        return self.preference.set_mode(mode)

    def get_preference_mode(self) -> str:
        return self.preference.mode

    def get_current_weights(self) -> Dict[str, float]:
        return self.preference.get_weights()
