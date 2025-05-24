import radar.domain as domain
import radar.algorithm as algorithm
import radar.experiments as experiments
import radar.data as data
import radar.utils as utils
import sys
from settings import params, nr_steps

# 获取算法名称和域名
params["algorithm_name"] = sys.argv[1]
params["domain_name"] = utils.get_argument(sys.argv, 2, None)
assert params["domain_name"] is not None, "domain_name is required"

# 对于BELIEF_QMIX，不需要对抗者比例参数
if params["algorithm_name"] == "BELIEF_QMIX":
    params["adversary_ratio"] = None  # 设置为None表示使用信念网络动态判断
    print("BELIEF_QMIX使用信念网络动态判断对抗者，无需设置对抗者比例")
else:
    # 其他算法仍然需要对抗者比例参数
    params["adversary_ratio"] = utils.get_float_argument(sys.argv, 3, None)
    assert params["adversary_ratio"] is not None, "对抗者比例参数是必需的"

params["test_suite"] = None
params["nr_test_episodes"] = 1

# 创建环境
env = domain.make(params["domain_name"], params)
nr_episodes = int(nr_steps/env.time_limit)

# 设置输出目录
if params["algorithm_name"] == "BELIEF_QMIX":
    params["directory"] = "{}/{}_{}".\
        format(params["test_directory"], 
               params["domain_name"], params["algorithm_name"])
else:
    params["directory"] = "{}/{}-agents_domain-{}_adversaryratio-{}_{}".\
        format(params["test_directory"], params["nr_agents"], 
               params["domain_name"], params["adversary_ratio"], 
               params["algorithm_name"])

params["directory"] = data.mkdir_with_timestap(params["directory"])

# 设置环境相关参数
params["global_observation_shape"] = env.global_observation_space.shape
params["local_observation_shape"] = env.local_observation_space.shape
params["state_shape"] = env.global_observation_space.shape  # 全局状态形状
params["obs_shape"] = env.local_observation_space.shape    # 局部观察形状
params["action_space"] = env.action_space                  # 动作空间
params["nr_actions"] = env.action_space.n
params["gamma"] = env.gamma
params["env"] = env

# 创建控制器
controller = algorithm.make(params["algorithm_name"], params)

# 运行实验
result = experiments.run(controller, nr_episodes, params, log_level=0)
