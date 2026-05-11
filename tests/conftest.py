"""
pytest conftest.py - CI/CD 适配配置
确保测试可以在 CI 环境中运行（使用环境变量配置 Redis）
"""

import os
import sys
import pytest
import redis

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope='session')
def redis_client():
    """
    Redis 客户端 fixture
    支持环境变量配置（CI 环境）
    """
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_db = int(os.getenv('REDIS_DB', 0))
    redis_password = os.getenv('REDIS_PASSWORD', None)
    
    try:
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False,  # 保持二进制模式（pickle 兼容）
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # 验证连接
        client.ping()
        print(f"✅ Redis 连接成功: {redis_host}:{redis_port}")
        return client
        
    except Exception as e:
        print(f"⚠️  Redis 连接失败: {e}")
        print("   使用 Mock Redis 继续测试...")
        return None


class MockRedis:
    """模拟 Redis 客户端（用于单元测试）"""
    def __init__(self):
        self.data = {}
    
    def get(self, key):
        return self.data.get(key)
    
    def set(self, key, value):
        self.data[key] = value
    
    def ping(self):
        return True
    
    def delete(self, key):
        if key in self.data:
            del self.data[key]
        return True


@pytest.fixture(scope='function')
def mock_redis():
    """每个测试函数使用独立的 Mock Redis"""
    return MockRedis()


@pytest.fixture(scope='session')
def config():
    """测试配置 fixture"""
    return {
        "models": {
            "enabled": [
                {"name": "hy3"},
                {"name": "qwen"},
                {"name": "gemma"}
            ],
            "fallback": "hy3"
        },
        "routing": {
            "task_type_mapping": {
                "code": ["hy3", "qwen"],
                "text": ["qwen", "gemma"],
                "default": ["hy3"]
            }
        }
    }


def pytest_configure(config):
    """pytest 配置（添加自定义标记）"""
    config.addinivalue_line(
        "markers",
        "slow: 标记慢速测试（性能基准、数值稳定性）"
    )
    config.addinivalue_line(
        "markers",
        "performance: 性能基准测试"
    )
    config.addinivalue_line(
        "markers",
        "stability: 数值稳定性测试"
    )


def pytest_collection_modifyitems(config, items):
    """根据命令行参数跳过特定测试"""
    # 如果没有指定 -m 参数，默认跳过 slow 测试
    if not config.getoption('-m'):
        skip_slow = pytest.mark.skip(reason="默认跳过慢速测试，使用 -m slow 启用")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
