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
nr_steps = 100000
# nr_steps = 500

params = {}
# 设置具体的对抗性智能体比例进行测试
params["test_adversary_ratios"] = [0.25, 0.5]  # 设置25%和50%的对抗性智能体
params["monitor_adversarial"] = True  # 启用对抗性智能体监控
params["adversarial_stats"] = {
    "log_frequency": 10,  # 每10步记录一次
    "track_values": True,  # 追踪智能体值
    "track_actions": True  # 追踪智能体行动
}

# 打印对抗性智能体的比例信息
for ratio in params["test_adversary_ratios"]:
    total_agents = 100  # 假设总智能体数量为100
    adversarial_count = int(ratio * total_agents)
    friendly_count = total_agents - adversarial_count
    logging.info(f"\n对抗性智能体配置:")
    logging.info(f"总智能体数量: {total_agents}")
    logging.info(f"对抗性智能体比例: {ratio:.2%}")
    logging.info(f"对抗性智能体数量: {adversarial_count}")
    logging.info(f"友好智能体数量: {friendly_count}")
    logging.info("-" * 50)

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