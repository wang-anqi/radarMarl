import radar.agents.controller as controller
import radar.agents.dqn as dqn
import radar.agents.ppo as ppo
import radar.agents.a2c as a2c
import radar.agents.a2cmix as a2cmix
import radar.agents.coma as coma
import radar.agents.maddpg as maddpg
import radar.agents.ppomix as ppomix
import radar.agents.vdn as vdn
import radar.agents.qmix as qmix
from radar.agents.belief_qmix import BeliefQMIXLearner

def make(algorithm, params={}):
    """创建算法实例
    
    参数:
        algorithm: 算法名称
        params: 算法参数字典
    
    返回:
        algorithm_instance: 算法实例
    """
    
    if algorithm == "BELIEF_QMIX":
        # 设置信念QMIX的默认参数
        params.setdefault("mixing_embed_dim", 32)     # 混合网络嵌入维度
        params.setdefault("hypernet_embed", 64)       # 超网络嵌入维度
        params.setdefault("belief_threshold", 0.6)    # 信念阈值
        params.setdefault("belief_update_freq", 10)   # 信念更新频率
        params.setdefault("hidden_sizes", [64, 64])   # 隐藏层大小
        params.setdefault("activation_func", "ReLU")  # 激活函数
        params.setdefault("use_recurrent_belief", True) # 是否使用循环信念
        params.setdefault("recurrent_N", 1)           # 循环层数
        params.setdefault("initialization_method", "orthogonal") # 初始化方法
        params.setdefault("gain", 0.01)               # 初始化增益
        params.setdefault("learning_rate", 0.001)     # 学习率
        params.setdefault("belief_learning_rate", 0.001) # 信念网络学习率
        params.setdefault("target_update_interval", 200) # 目标网络更新间隔
        params.setdefault("memory_capacity", 20000)   # 经验回放容量
        params.setdefault("batch_size", 32)           # 批次大小
        params.setdefault("gamma", 0.99)              # 折扣因子
        
        print("\n" + "="*50)
        print("初始化信念QMIX算法:")
        print("-"*50)
        print(f"混合网络嵌入维度: {params['mixing_embed_dim']}")
        print(f"超网络嵌入维度: {params['hypernet_embed']}")
        print(f"信念阈值: {params['belief_threshold']}")
        print(f"信念更新频率: {params['belief_update_freq']}")
        print(f"隐藏层大小: {params['hidden_sizes']}")
        print(f"是否使用循环信念: {params['use_recurrent_belief']}")
        print(f"学习率: {params['learning_rate']}")
        print(f"信念网络学习率: {params['belief_learning_rate']}")
        print("="*50 + "\n")
        
        return BeliefQMIXLearner(params)
    
    if algorithm == "BELIEF_RADAR":
        # 设置belief相关的默认参数
        params.setdefault("belief_threshold", 0.6)
        params.setdefault("min_adversaries", 1)
        params.setdefault("max_adversaries", params["nr_agents"] - 1)
        params.setdefault("hidden_sizes", [64, 64])
        params.setdefault("activation_func", "ReLU")
        params.setdefault("use_recurrent_belief", True)
        params.setdefault("recurrent_N", 1)
        params.setdefault("initialization_method", "orthogonal")
        params.setdefault("gain", 0.01)
        
        print("\n" + "="*50)
        print("初始化BELIEF_RADAR算法:")
        print("-"*50)
        print(f"信念阈值: {params['belief_threshold']}")
        print(f"最小对抗者数量: {params['min_adversaries']}")
        print(f"最大对抗者数量: {params['max_adversaries']}")
        print("="*50 + "\n")
        
        from radar.agents.belief_controller import BeliefController
        return BeliefController(params)
    
    if algorithm == "Random":
        params["adversary_ratio"] = 0
        return controller.Controller(params)
        
    if algorithm == "DQN":
        params["adversary_ratio"] = 0
        return dqn.DQNLearner(params)
        
    if algorithm == "PPO":
        params["adversary_ratio"] = 0
        return ppo.PPOLearner(params)
        
    if algorithm == "IAC":
        params["adversary_ratio"] = 0
        return a2c.A2CLearner(params)
        
    if algorithm == "RAT_IAC":
        params["adversary_ratio"] = None
        return a2c.A2CLearner(params)
        
    if algorithm == "AC-QMIX":
        params["adversary_ratio"] = 0
        params["central_q_learner"] = qmix.QMIXLearner(params)
        return a2cmix.A2CMIXLearner(params)
        
    if algorithm == "RADAR_X":
        params["central_q_learner"] = vdn.VDNLearner(params)
        return a2cmix.A2CMIXLearner(params)
        
    if algorithm == "RADAR":
        params["adversary_ratio"] = None
        params["central_q_learner"] = vdn.VDNLearner(params)
        return a2cmix.A2CMIXLearner(params)
        
    if algorithm == "COMA":
        params["adversary_ratio"] = 0
        return coma.COMALearner(params)
        
    if algorithm == "MADDPG":
        params["minimax"] = False
        assert params["adversary_ratio"] is not None, "MADDPG需要指定adversary_ratio参数"
        return maddpg.MADDPGLearner(params)
        
    if algorithm == "M3DDPG":
        params["minimax"] = True
        params["adversary_ratio"] = 0  # 对抗者在Q函数中建模
        return maddpg.MADDPGLearner(params)
        
    if algorithm == "PPO-QMIX":
        params["adversary_ratio"] = 0
        params["central_q_learner"] = qmix.QMIXLearner(params)
        return ppomix.PPOMIXLearner(params)
        
    if algorithm == "RADAR_PPO":
        params["adversary_ratio"] = None
        params["central_q_learner"] = vdn.VDNLearner(params)
        return ppomix.PPOMIXLearner(params)
        
    if algorithm == "RAT_PPO":
        params["adversary_ratio"] = None
        return ppo.PPOLearner(params)
        
    if algorithm == "RAT_DQN":
        params["adversary_ratio"] = None
        return dqn.DQNLearner(params)
        
    if algorithm == "VDN":
        params["adversary_ratio"] = 0
        return vdn.VDNLearner(params)
        
    if algorithm == "QMIX":
        params["adversary_ratio"] = 0
        return qmix.QMIXLearner(params)
        
    raise ValueError(f"未知的算法: '{algorithm}'")