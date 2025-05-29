import logging
import sys

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# nr_steps = 2000000
# nr_steps = 100000
nr_steps = 200

params = {}
# 移除手动设置的对抗性智能体比例
params["monitor_adversarial"] = True  # 启用对抗性智能体监控
params["adversarial_stats"] = {
    "log_frequency": 10,  # 每10步记录一次
    "track_values": True,  # 追踪智能体值
    "track_actions": True  # 追踪智能体行动
}

# 保持原有的其他参数
params["test_algorithms"] = ["RADAR_X"]
params["test_directory"] = "tests"
params["test_interval"] = 10
params["nr_test_episodes"] = 50
params["use_global_reward"] = True
params["save_summaries"] = True
params["alpha"] = 0.01

# These hyperparameters are only required for DQN, VDN, QMIX
params["warmup_phase"] = 5000
params["target_update_period"] = 4000
params["memory_capacity"] = 20000
params["epsilon_decay"] = 1.0/50000

# Uncomment to manually set random seed
"""
GLOBAL_SEED = 42
import torch
import numpy
import random
torch.manual_seed(GLOBAL_SEED)
numpy.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
"""

params["time_limit"] = 100  # 增加每个episode的时间限制
params["view_range"] = 7    # 增加智能体的视野范围