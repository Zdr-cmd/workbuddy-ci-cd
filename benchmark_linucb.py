"""
性能对比测试：Sherman-Morrison 增量更新 (O(d²)) vs 直接求逆 (O(d³))
"""
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from core.linucb_router import LinUCBRouter


class MockRedis:
    """模拟 Redis 客户端"""
    def __init__(self):
        self.data = {}
    
    def get(self, key):
        return self.data.get(key)
    
    def set(self, key, value):
        self.data[key] = value
    
    def ping(self):
        return True


def benchmark_incremental_update():
    """基准测试：增量更新 O(d²)"""
    d = 12
    n_updates = 10000
    
    config = {"models": {"enabled": [{"name": "hy3"}]}}
    router = LinUCBRouter(MockRedis(), config, feature_dim=d, alpha=1.0)
    
    # 预热：确保矩阵已初始化
    router.update("hy3", {"task_type": "code", "input_length": 100}, reward=0.8)
    
    # 基准测试
    start = time.time()
    for i in range(n_updates):
        context = {
            "task_type": "code",
            "input_length": 100 + i,
            "hour_of_day": i % 24
        }
        router.update("hy3", context, reward=np.random.rand())
    
    elapsed = time.time() - start
    print(f"✅ 增量更新 O(d²) 耗时: {elapsed:.3f}s ({n_updates}次更新)")
    print(f"   平均每次: {elapsed/n_updates*1000:.3f}ms")
    
    return elapsed


def benchmark_direct_inv():
    """基准测试：直接求逆 O(d³)"""
    d = 12
    n_updates = 10000
    
    A = np.identity(d)
    b = np.zeros(d)
    
    start = time.time()
    for i in range(n_updates):
        x = np.random.rand(d)
        A += np.outer(x, x)
        b += 0.8 * x
        A_inv = np.linalg.inv(A)  # O(d³) - 每次都重新求逆
    
    elapsed = time.time() - start
    print(f"⚠️  直接求逆 O(d³) 耗时: {elapsed:.3f}s ({n_updates}次更新)")
    print(f"   平均每次: {elapsed/n_updates*1000:.3f}ms")
    
    return elapsed


def benchmark_select_arm():
    """基准测试：模型选择（使用增量维护的 A_inv）"""
    d = 12
    n_selections = 10000
    
    config = {"models": {"enabled": [{"name": "hy3"}, {"name": "qwen"}]}}
    router = LinUCBRouter(MockRedis(), config, feature_dim=d, alpha=1.0)
    
    # 预热
    router.update("hy3", {"task_type": "code", "input_length": 100}, reward=0.8)
    router.update("qwen", {"task_type": "code", "input_length": 100}, reward=0.7)
    
    # 基准测试
    start = time.time()
    for i in range(n_selections):
        context = {
            "task_type": "code",
            "input_length": 100 + i
        }
        router.select_arm(context)
    
    elapsed = time.time() - start
    print(f"✅ 模型选择 O(d²) 耗时: {elapsed:.3f}s ({n_selections}次选择)")
    print(f"   平均每次: {elapsed/n_selections*1000:.3f}ms")
    
    return elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("LinUCB 性能基准测试")
    print("=" * 60)
    print()
    
    # 测试1：增量更新 vs 直接求逆
    print("【测试1】更新性能对比")
    print("-" * 60)
    t_inc = benchmark_incremental_update()
    print()
    t_dir = benchmark_direct_inv()
    print()
    
    speedup = t_dir / t_inc if t_inc > 0 else float('inf')
    print(f"🚀 加速比: {speedup:.1f}x")
    print()
    
    # 测试2：模型选择性能
    print("【测试2】模型选择性能")
    print("-" * 60)
    t_sel = benchmark_select_arm()
    print()
    
    print("=" * 60)
    print("✅ 基准测试完成！")
    print("=" * 60)
