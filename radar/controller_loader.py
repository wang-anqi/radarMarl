import radar.algorithm as algorithm
import radar.data as data
from os.path import join
import radar.utils as utils
import random
import os

def get_paths(basepath, algorithm_name, params):
    """获取模型路径
    
    参数:
        basepath: 基础路径
        algorithm_name: 算法名称
        params: 参数字典
    """
    # 对于 BELIEF_QMIX，使用简化的路径模式
    if algorithm_name == "BELIEF_QMIX":
        data_prefix_pattern = "{}_{}_".format(
            params["domain_name"],
            algorithm_name
        )
    else:
        # 其他算法使用原有的路径模式
        adversary_ratio = params["adversary_ratio"]
        if adversary_ratio is not None:
            adversary_ratio = float(adversary_ratio)
        data_prefix_pattern = "{}-agents_domain-{}_adversaryratio-{}_{}_".format(
            params["nr_agents"],
            params["domain_name"], 
            adversary_ratio, 
            algorithm_name
        )
    
    # 如果目录不存在，创建它
    if not os.path.exists(basepath):
        os.makedirs(basepath)
        print(f"创建目录: {basepath}")
    
    directories = data.list_directories(basepath, lambda x: x.startswith(data_prefix_pattern))
    result = []
    predicate = lambda a,x: x.startswith("protagonist_model") or x.startswith("adversary_model")
    for directory in directories:
        if len(data.list_files_with_predicate(directory, predicate)) == 2:
            result.append(directory)
    
    # 如果没有找到任何路径，返回空列表而不是抛出错误
    if len(result) == 0:
        print(f"警告: 未找到匹配的模型路径 {data_prefix_pattern}")
        return []
        
    return result

def load_agents(path, algorithm_name, params):
    agents = algorithm.make(algorithm_name, params)
    agents.load_weights(path)
    return agents

def combine_agents(protagonist_agents, adversary_agents, algorithm_name, params):
    agents = algorithm.make(algorithm_name, params)
    agents.policy_net.protagonist_net = protagonist_agents.policy_net.protagonist_net
    agents.policy_net.adversary_net = adversary_agents.policy_net.adversary_net
    return agents