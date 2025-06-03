import radar.domain as domain
import radar.algorithm as algorithm
import radar.experiments as experiments
import radar.data as data
import radar.utils as utils
import sys
from settings import params, nr_steps
from os.path import join

def print_usage():
    """打印使用说明"""
    print("\n使用方法:")
    print("python training_run.py <算法名称> <域名> [对抗者比例]")
    print("\n参数说明:")
    print("- 算法名称: 必需参数")
    print("- 域名: 必需参数")
    print("- 对抗者比例: 对BELIEF_QMIX可选，其他算法必需 (0-1之间的浮点数)")
    print("\n示例:")
    print("python training_run.py QMIX domain_name 0.5")
    print("python training_run.py BELIEF_QMIX domain_name")
    sys.exit(1)

# 检查基本参数数量
if len(sys.argv) < 3:
    print("\n错误: 参数不足")
    print_usage()

# 获取算法名称
params["algorithm_name"] = sys.argv[1]

# 添加历史特征相关参数
if params["algorithm_name"] == "BELIEF_QMIX":
    print("\n使用完整历史信息进行智能体行为分析")

# 获取域名
params["domain_name"] = utils.get_argument(sys.argv, 2, None)
if params["domain_name"] is None:
    print("\n错误: 必须提供域名参数")
    print_usage()

# 处理对抗者比例参数
if params["algorithm_name"] == "BELIEF_QMIX":
    params["adversary_ratio"] = None
    print("\n提示: BELIEF_QMIX使用信念网络动态判断对抗者，无需设置对抗者比例")
else:
    params["adversary_ratio"] = utils.get_float_argument(sys.argv, 3, None)
    if params["adversary_ratio"] is None:
        print(f"\n错误: {params['algorithm_name']} 算法需要提供对抗者比例参数")
        print_usage()
    # 验证对抗者比例是否在有效范围内
    if not (0 <= params["adversary_ratio"] <= 1):
        print("\n错误: 对抗者比例必须在0到1之间") 
        print_usage()

params["test_suite"] = experiments.run_test_suite
params["nr_test_episodes"] = 50

# 创建环境
try:
    env = domain.make(params["domain_name"], params)
except Exception as e:
    print(f"\n错误: 创建环境失败 - {str(e)}")
    sys.exit(1)

nr_episodes = int(nr_steps/env.time_limit)
params["nr_steps"] = nr_steps

# 设置输出目录
if params["algorithm_name"] == "BELIEF_QMIX":
    params["directory"] = "output/{}_{}".\
        format(params["domain_name"], params["algorithm_name"])
else:
    params["directory"] = "output/{}-agents_domain-{}_adversaryratio-{}_{}".\
        format(params["nr_agents"], params["domain_name"],\
            params["adversary_ratio"], params["algorithm_name"])

params["directory"] = data.mkdir_with_timestap(params["directory"])

# 设置环境相关参数
params["global_observation_shape"] = env.global_observation_space.shape
params["local_observation_shape"] = env.local_observation_space.shape
params["state_shape"] = env.global_observation_space.shape
params["obs_shape"] = env.local_observation_space.shape
params["action_space"] = env.action_space
params["nr_actions"] = env.action_space.n
params["gamma"] = env.gamma
params["env"] = env

print("\n训练配置:")
print(f"总训练步数: {nr_steps}")
print(f"每轮步数: {env.time_limit}")
print(f"预计训练轮数: {nr_episodes}")
print(f"算法: {params['algorithm_name']}")
print(f"环境: {params['domain_name']}")
if params["adversary_ratio"] is not None:
    print(f"对抗者比例: {params['adversary_ratio']}")
print()

# 创建控制器
try:
    controller = algorithm.make(params["algorithm_name"], params)
except Exception as e:
    print(f"\n错误: 创建控制器失败 - {str(e)}")
    sys.exit(1)

# 运行实验
try:
    result = experiments.run(controller, nr_episodes, params, log_level=0)
except Exception as e:
    print(f"\n错误: 实验运行失败 - {str(e)}")
    sys.exit(1)

# 保存结果
data.save_json(join(params["directory"], "returns.json"), result)
print(f"\n训练完成! 实际训练步数: {result['total_steps']}")
print(f"结果已保存到: {params['directory']}")
