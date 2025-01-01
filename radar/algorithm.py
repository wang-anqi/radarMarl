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

def make(algorithm, params={}):
    """创建算法实例"""
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
        print("Initializing BELIEF_RADAR Algorithm:")
        print("-"*50)
        print(f"Belief threshold: {params['belief_threshold']}")
        print(f"Min adversaries: {params['min_adversaries']}")
        print(f"Max adversaries: {params['max_adversaries']}")
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
        # assert params["adversary_ratio"] is not None, "RADAR (X), requires adversary-ratio as float"
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
        assert params["adversary_ratio"] is not None, "MADDPG, requires adversary-ratio as float"
        return maddpg.MADDPGLearner(params)
    if algorithm == "M3DDPG":
        params["minimax"] = True
        params["adversary_ratio"] = 0 # Adversaries are modeled within Q-function
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
    raise ValueError("Unknown algorithm '{}'".format(algorithm))