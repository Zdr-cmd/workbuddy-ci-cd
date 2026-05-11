# WorkBuddy 运行章程 v5.0.0（正式版）

**版本**: 5.0.0
**发布日期**: 2026-05-11
**基于**: v4.4.0（增强务实版）+ ML高级功能 + 合规框架
**状态**: ✅ 正式发布

---

## 版本说明

**v4.4.0 → v5.0.0 重大升级**：

### 新增核心功能
1. **LinUCB 上下文 Bandit 路由**（第14章）— Sherman-Morrison 增量更新，Redis 持久化
2. **ML 告警自动降级**（第4.3节）— 三种告警类型自动处理（累积后悔/预测误差/特征漂移）
3. **因果推断框架**（第15.3节）— 双重差分（DiD）评估进化动作因果效应
4. **MAML 元学习**（第15.4节）— 少样本任务适应，元网络快速更新
5. **合规保障框架**（第15.5节）— 规则化合规检查，运行时审计钩子
6. **ACP 模式 + 防御性遗忘**（第3.4-3.6节）— 安全操作与遗忘机制

### 预期收益
- LinUCB 路由相比静态路由预期提升 15-20% 任务成功率
- ML 告警自动处理减少 80% 人工干预
- 因果推断帮助识别真正有效的进化动作
- MAML 少样本适应冷启动时间缩短 50%
- 合规框架确保所有操作可审计、可回滚

### 迁移说明
- 第14章 LinUCB 为可选模块（通过 `BANDIT_MODE=true` 开启，默认关闭使用加权路由）
- 第15章因果推断/MAML 需要 `torch`, `statsmodels`, `scikit-learn` 依赖
- 第3.4-3.6节安全模块新增，存量代码不受影响

---

## 0. 元规则：章程生命周期

### 0.1 版本号语义
- **主版本**（5.0）：核心架构变更，影响多个子系统
- **次版本**（5.x）：新增功能，向后兼容
- **修订版本**（5.x.x）：Bug修复，文档更新

### 0.2 变更管理
- 任何章节变更必须通过版本对比工具确认影响范围
- 新增章节需通过 `on_ml_alert` 等核心接口的集成测试
- 涉及破坏性变更须创建 Migration Guide

### 0.3 元参数

```python
charter_version = "5.0.0"
acp_mode = True
evolution_freq = 20
rate_limit = 5
bandit_mode = False  # 默认关闭，v5.1+ 开启
compliance_enabled = True  # 默认开启
```

---

## 1. 不可变更的核心原则

> 以下原则在任何版本、任何条件下均不可被覆盖或临时禁用。

1. **工作记忆预算** — `claude-code` 的最大 token 预算（200K）不可突破
2. **安全边界** — `!!HALT` 立即停止所有操作，无条件执行
3. **进化冻结** — 当 `freeze_evolution()` 被调用时，所有自动调整暂停
4. **可审计性** — 所有外部操作必须记录到 `audit.log`
5. **宪章优先** — 当用户指令与宪章冲突时，宪章优先，提示用户

---

## 2. 高效运行规则

### 2.1 工作记忆预算

（同 v4.4.0，保持不变）

### 2.2 递归分解

（同 v4.4.0，保持不变）

### 2.3 Skill 缓存与重用（语义缓存版）

（同 v4.4.0，保持不变）

### 2.4 异步批处理

（同 v4.4.0，保持不变）

### 2.5 模型路由（第14章 LinUCB 版 — 新增）

**不使用 LinUCB** 时，使用简化版加权路由（v4.4.0 第2.5节）。**开启 LinUCB** 时，优先使用第14章实现。

#### 2.5.1 降级策略

（同 v4.4.0，保持不变）

#### 2.5.2 用户覆盖

（同 v4.4.0，保持不变）

---

## 3. 安全防护体系

（同 v4.4.0 第3.1-3.3节，保持不变）

### 3.4 ACP 模式（新增）

ACP（Adversarial Condition Prevention）模式：检测并阻止潜在对抗性操作。

```python
class ACPMode:
    def __init__(self):
        self.enabled = True
        self.threat_signatures = [
            "prompt_injection",
            "system_override",
            "context_manipulation"
        ]

    def check(self, action: str, context: dict) -> bool:
        """
        返回 True 表示安全，False 表示阻止
        """
        if not self.enabled:
            return True
        # 检测 prompt injection 模式
        if any(kw in str(context) for kw in self.threat_signatures):
            return False
        return True
```

### 3.5 三明治检查（新增）

所有 Skill 执行前后进行三明治式检查：

```python
def sandwich_check(skill_name: str, input_data: dict, output_data: dict) -> tuple:
    """
    返回 (is_safe, reason)
    - 前置检查：Skill 签名、输入合法性
    - 后置检查：输出完整性、无敏感泄露
    """
    # 前置
    if not verify_skill_signature(skill_name):
        return False, "Skill 签名验证失败"
    if not verify_input_schema(input_data):
        return False, "输入格式不符合 SKILL.md 规范"

    # 后置
    if contains_sensitive_leak(output_data):
        return False, "输出包含敏感信息泄露"
    if not verify_output_completeness(output_data):
        return False, "输出不完整或格式异常"

    return True, "OK"
```

### 3.6 防御性遗忘（新增）

三级遗忘响应机制：

| 级别 | 触发条件 | 遗忘范围 | 用户通知 |
|------|---------|---------|---------|
| L1-临时 | 短时异常（< 1h） | 当前会话记忆 | 无 |
| L2-上下文 | 中时异常（1-24h） | 相关上下文片段 | 提示"记忆丢失" |
| L3-深度 | 严重安全事件 | 全局记忆 + 技能缓存 | 完整告警 |

```python
class DefensiveForgetting:
    def trigger(self, level: int, reason: str):
        if level == 1:
            self._forget_session()
        elif level == 2:
            self._forget_contextual()
        elif level == 3:
            self._forget_deep(reason)
        self.log_forgetting_event(level, reason)
```

---

## 4. 自动进化框架

### 4.1-4.2 （同 v4.4.0，保持不变）

### 4.3 ML 告警自动降级（新增 — v5.0.0 核心功能）

当 ML 监控系统检测到 LinUCB 路由性能下降时，自动执行降级或调参，无需人工干预。

#### 4.3.1 三种告警类型

| 告警类型 | 触发指标 | 自动动作 |
|---------|---------|---------|
| `bandit_cumulative_regret` | 累积后悔持续增长超过阈值 | 关闭 Bandit，降级为加权路由；冻结进化24h |
| `model_prediction_error` | 预测误差 MAE > 阈值 | 增加探索率 ε（每次+0.05，上限0.5） |
| `feature_distribution_shift` | KL散度/KS检验漂移 | 通知用户重训特征编码器 |

#### 4.3.2 on_ml_alert() 实现

```python
def on_ml_alert(alert_type: str, metric_value: float, threshold: float):
    """
    ML 告警处理函数（自动降级/调参）
    由 AlertManager 或 ml_monitor 在检测到告警时调用
    """
    if is_evolution_frozen():
        print(f"[Evolution] 进化已冻结，忽略 ML 告警 {alert_type}")
        return

    if alert_type == "bandit_cumulative_regret":
        record_failure("ml_bandit",
                       f"累积后悔 {metric_value} 超阈值 {threshold}", "自动降级")
        set_bandit_mode(False)              # 关闭 LinUCB
        freeze_evolution(duration_hours=24)  # 暂停进化24h
        send_user_notification(
            "⚠️ 模型效果下降，已自动切换为保守路由策略。"
        )

    elif alert_type == "model_prediction_error":
        current_eps = get_bandit_epsilon()
        new_eps = min(current_eps + 0.05, 0.5)
        set_bandit_epsilon(new_eps)
        send_user_notification(
            f"🔍 预测误差偏高 (MAE={metric_value:.3f})，"
            f"已提高探索率至 {new_eps:.0%}"
        )

    elif alert_type == "feature_distribution_shift":
        print(f"[Evolution] 检测到特征漂移 KL={metric_value:.3f}")
        send_user_notification(
            "📊 检测到输入特征分布发生变化，建议重新训练特征编码器。"
        )

    else:
        print(f"[Evolution] 未知 ML 告警类型: {alert_type}")
```

#### 4.3.3 全局状态函数

```python
_bandit_mode = True           # 当前是否启用 Bandit 路由（默认开启）
_bandit_epsilon = 0.1         # 当前探索率 ε

def set_bandit_mode(enabled: bool) -> bool: ...
def set_bandit_epsilon(epsilon: float) -> bool: ...
def get_bandit_mode() -> bool: ...
def get_bandit_epsilon() -> float: ...
def is_evolution_frozen() -> bool: ...
```

---

## 5. 沟通协议

（同 v4.4.0，保持不变）

---

## 6. 记忆与遗忘

（同 v4.4.0，保持不变）

---

## 7. 观测仪表盘（ML监控增强 - 简化版）

（同 v4.4.0 第7.1-7.6节，保持不变）

---

## 8. 启动初始化

（同 v4.4.0，保持不变；额外增加 LinUCB 初始化）

```python
# 新增启动初始化
from core.linucb_router import create_linucb_router

def initialize_linucb(config):
    """启动时初始化 LinUCB 路由器"""
    redis_url = config.get("redis_url")
    return create_linucb_router(redis_url=redis_url, config=config)
```

---

## 9. 版本升级兼容性

（同 v4.4.0，保持不变）

---

## 10. 知识迁移与跨 Skill 学习

（同 v4.4.0 第10.1-10.5节，保持不变）

---

## 11. 在线实验框架（A/B测试）

（同 v4.4.0，保持不变）

---

## 12. 宪章测试体系（新增）

### 12.1 测试分类

| 测试级别 | 覆盖内容 | 执行频率 | 通过标准 |
|---------|---------|---------|---------|
| P0 | LinUCB 核心逻辑、ML 告警降级、合规检查 | 每次 PR | 100% 通过 |
| P1 | 模型路由策略切换、遗忘机制 | 每日 | 100% 通过 |
| P2 | 因果推断、MAML 训练 | 每周 | 100% 通过 |

### 12.2 P0 单元测试

```python
# tests/test_evolution_ml_alert.py — 覆盖三种 ML 告警类型

def test_bandit_cumulative_regret_alert():
    """触发 bandit_cumulative_regret 告警 → 降级为加权路由"""
    from core.evolution import on_ml_alert, get_bandit_mode
    on_ml_alert("bandit_cumulative_regret", metric_value=0.95, threshold=0.8)
    assert get_bandit_mode() is False

def test_model_prediction_error_alert():
    """触发 model_prediction_error 告警 → ε 增加"""
    from core.evolution import on_ml_alert, get_bandit_epsilon, set_bandit_epsilon
    set_bandit_epsilon(0.1)
    on_ml_alert("model_prediction_error", metric_value=0.15, threshold=0.1)
    assert get_bandit_epsilon() == 0.15  # 0.1 + 0.05

def test_feature_distribution_shift_alert():
    """触发 feature_distribution_shift 告警 → 通知用户"""
    from core.evolution import on_ml_alert
    # 不应降级或改变模式
    on_ml_alert("feature_distribution_shift", metric_value=0.5, threshold=0.3)
```

### 12.3 合规测试

```python
# tests/test_compliance.py

def test_safety_violation_blocked():
    """违反安全规则应被阻止"""
    from core.compliance import CharterCompliance
    c = CharterCompliance()
    with pytest.raises(RuleViolation):
        c.check("safety.1.1", {"action": "execute_native_process", "confirmed": False})

def test_evolution_frozen_blocks_alert():
    """进化冻结期不得触发自动调整"""
    from core.evolution import freeze_evolution, on_ml_alert, is_evolution_frozen
    freeze_evolution(duration_hours=1)
    assert is_evolution_frozen() is True
    on_ml_alert("bandit_cumulative_regret", 0.95, 0.8)
    # 告警被忽略，bandit_mode 不变
```

### 12.4 迁移脚本

```python
# scripts/migrate_v4_to_v5.py
"""v4.4.0 → v5.0.0 迁移脚本"""
import shutil, os

def migrate():
    # 1. 备份
    shutil.copy("charter.md", "backups/charter_v4.4.0.md")
    # 2. 添加新章节（追加到 charter.md）
    with open("charter.md", "a", encoding="utf-8") as f:
        f.write("\n## v5.0.0 新增章节\n")
        f.write("（见 CHARTER_v5.0.0.md）\n")
    # 3. 安装新依赖
    os.system("pip install torch statsmodels scikit-learn")
    print("迁移完成，重启 WorkBuddy 生效")
```

---

## 13. LinUCB 上下文 Bandit 路由（新增 — 第14章）

### 14.1 核心设计

- **算法**：LinUCB（Linear Upper Confidence Bound）
- **更新复杂度**：O(d²) — 使用 Sherman-Morrison 增量更新逆矩阵
- **持久化**：Redis 存储 A、b、A_inv 参数
- **特征维度**：默认 12 维，可配置
- **探索系数 α**：默认 1.0，可配置

### 14.2 核心算法

```python
# 选择 arm（模型）
theta = A_inv @ b
mean = theta @ x
std = sqrt(x @ A_inv @ x)
ucb = mean + alpha * std
best_arm = argmax(ucb)

# 更新（Sherman-Morrison）
A_new = A_old + x @ x^T
A_inv_new = A_inv_old - (A_inv_old @ x @ x^T @ A_inv_old) / (1 + x^T @ A_inv_old @ x)
b_new = b_old + reward * x
```

### 14.3 特征设计

| 维度 | 特征 | 说明 |
|------|------|------|
| 0-3 | 任务类型 one-hot | code_generation / long_text / creative / reasoning |
| 4 | 输入长度归一化 | min(len/4096, 1.0) |
| 5-6 | 时间周期特征 | sin/cos(hour/24) |
| 7 | 用户满意度均值 | 滑动平均 |
| 8-10 | 模型性能特征 | 成功率、成本、复杂度 |
| 11 | 扩展位 | 备用 |

### 14.4 自动重构机制

- 每 1000 次更新强制重新计算逆矩阵（`np.linalg.inv`）保证数值稳定性
- 检查条件数 > 1e10 时立即重构

### 14.5 降级与恢复

- 当 `bandit_cumulative_regret` 告警触发时，自动降级到加权路由（`WeightedModelRouter`）
- 恢复：`/ml bandit on` 命令手动开启

---

## 14. 因果推断框架（新增 — 第15.3节）

### 15.3.1 设计目标

- 评估进化动作的**真实因果效应**（而非相关性）
- 使用**双重差分（DiD）**比较实验组和对照组在动作前后的差异变化
- 支持**倾向性得分匹配（PSM）**处理选择性偏差

### 15.3.2 核心接口

```python
from core.causal_inference import CausalImpactEvaluator

evaluator = CausalImpactEvaluator(log_path="logs/audit.log")

# 评估某个进化动作的因果效应
result = evaluator.evaluate_evolution(
    action_name="model_selection_tweak",
    action_time=pd.Timestamp("2026-05-01"),
    metric="success"
)
# result: {"causal_effect": 0.12, "p_value": 0.03, "significant": True}
```

---

## 15. MAML 元学习（新增 — 第15.4节）

### 15.4.1 设计目标

- 通过**元学习**实现少样本任务适应
- 训练一个**元网络**，将任务特征映射为 LinUCB 参数
- 支持**内循环快速适应**（5步梯度更新）和**外循环元更新**

### 15.4.2 核心组件

| 组件 | 职责 |
|------|------|
| `LinUCBBase` | 可微分 LinUCB 基学习器（PyTorch 模块） |
| `MAMLMetaLearner` | 元网络：编码任务特征为 (A, b) 参数 |
| `train_maml()` | 外循环训练函数 |

### 15.4.3 训练接口

```python
from core.meta_learner import train_maml, LinUCBBase, MAMLMetaLearner
import torch

# 准备元任务数据
meta_tasks = [
    {"task_features": torch.randn(10, 8), "rewards": torch.rand(10)}
    for _ in range(50)
]

# 训练 MAML
meta_learner = train_maml(
    meta_tasks=meta_tasks,
    feature_dim=12,
    task_feature_dim=8,
    epochs=100
)
```

---

## 15. 合规保障框架（新增 — 第15.5节）

### 15.5.1 核心设计

- 所有操作必须通过 `CharterCompliance.check()` 验证
- 违规时触发 `RuleViolation` 异常并记录
- 合规日志持久化到 `logs/compliance.log`

### 15.5.2 规则表

| 规则 ID | 描述 | 检查条件 |
|--------|------|---------|
| safety.1.1 | 禁止未确认外部操作 | confirmed=True |
| safety.1.2 | 禁止删除 .workbuddy | 路径不含 .workbuddy |
| safety.1.3 | 禁止递归删除个人目录 | 非递归或非个人路径 |
| evolution.2.1 | 冻结期不得自动调整 | force=True |
| evolution.2.2 | ML 降级必须记录 | recorded=True |
| memory.3.1 | 敏感信息不得写 MEMORY.md | sensitive≠True |

### 15.5.3 使用接口

```python
from core.compliance import check_rule, CharterCompliance

c = CharterCompliance()

# 检查单条规则
c.check("safety.1.1", {"action": "execute_native_process", "confirmed": False})
# → 触发 RuleViolation

# 检查所有规则
failed = c.check_all(context)
print(f"失败规则: {failed}")

# 合规报告
report = c.get_compliance_report()
```

---

## 16. 附录

### 附录A：已验证能力清单

（同 v4.4.0，增加 `torch`, `statsmodels`, `scikit-learn`, `redis` 依赖）

---

### 附录B：命令速查表（v5.0.0）

| 命令 | 作用 |
|------|------|
| `/status` | 仪表盘 |
| `/status drift` | 漂移检测报告 |
| `/good <score>` | 评分 1~5 |
| `/model <name>` | 临时切换模型 |
| `/model reset` | 恢复动态路由 |
| `/cost breakdown` | 成本细粒度归因 |
| `/ml bandit on` | 开启 LinUCB Bandit |
| `/ml bandit off` | 关闭 Bandit，降级到加权路由 |
| `/compliance report` | 合规报告 |
| `!!HALT` | 紧急停止 |
| `/alerts` | 查看活跃告警 |
| `/memorize` | 永久记忆 |
| `/lock_evolution` | 暂停进化 |
| `/reset_safety` | 退出隔离模式 |
| `/rollback` | 回滚到上一稳定版本 |

---

### 附录C：依赖清单

```
# v5.0.0 新增依赖
torch>=2.0.0        # MAML 元学习
statsmodels>=0.14.0 # 因果推断（DiD 回归）
scikit-learn>=1.3.0 # PSM 倾向性得分匹配
redis>=5.0.0        # LinUCB 参数持久化
```

---

### 附录D：Skill 编写规范

（同 v4.4.0 附录D）

---

### 附录E：知识迁移决策树

（同 v4.4.0 附录E）

---

### 附录F：成本分解详细格式

（同 v4.4.0 附录F）

---

## 文档控制

- **批准者**: 用户
- **下次全面复盘**: 实施后3个月
- **版本历史**:
  - v4.0: 初始实际架构版
  - v4.1: 冲突解决、动态路由、自动采集
  - v4.2: 工程精简版（移除过度设计）
  - v4.3: 生产就绪版
  - v4.4: 可行改进版（语义缓存、成本归因）
  - **v5.0**: 智能进化版（LinUCB + ML告警 + 因果推断 + MAML + 合规框架）

---

**文档结束**
