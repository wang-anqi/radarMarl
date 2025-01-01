import random
import numpy
import torch
from radar.utils import get_param_or_default
from radar.utils import pad_or_truncate_sequences
from radar.belief.belief import Belief

class ReplayMemory:

    def __init__(self, capacity, is_prioritized=False):
        self.transitions = []
        self.capacity = capacity
        self.nr_transitions = 0

    def save(self, transition):
        self.transitions.append(transition)
        self.nr_transitions += len(transition[0])
        if self.nr_transitions > self.capacity:
            removed_transition = self.transitions.pop(0)
            self.nr_transitions -= len(removed_transition[0])

    def sample_batch(self, minibatch_size):
        nr_episodes = self.size()
        if nr_episodes > minibatch_size:
            return random.sample(self.transitions, minibatch_size)
        return self.transitions

    def clear(self):
        self.transitions.clear()
        self.nr_transitions = 0

    def size(self):
        return len(self.transitions)

class Controller:

    def __init__(self, params):
        # 初始化基本参数
        self.nr_agents = params["nr_agents"]
        self.nr_actions = params["nr_actions"]
        self.actions = list(range(self.nr_actions))
        
        # 初始化belief网络及其状态
        self.belief_net = Belief(params, params["local_observation_shape"], 
                               params["nr_actions"], params["nr_agents"], 
                               device=torch.device("cpu"))
        self.belief_rnn_states = torch.zeros(params["nr_agents"], 
                                           params["hidden_sizes"][-1])
        self.masks = torch.ones(params["nr_agents"], 1)
        
        # 初始化对抗者ID列表
        self.adversary_ids = []
        
        # 设置belief相关参数
        self.belief_threshold = get_param_or_default(params, "belief_threshold", 0.6)  # 对抗倾向阈值
        self.min_adversaries = get_param_or_default(params, "min_adversaries", 1)  # 最小对抗者数量
        self.max_adversaries = get_param_or_default(params, "max_adversaries", self.nr_agents - 1)  # 最大对抗者数量
        
        # 设置其他参数
        self.gamma = params["gamma"]
        self.alpha = get_param_or_default(params, "alpha", 0.001)
        self.env = params["env"]

    def generate_adversary_ids(self, is_adversary):
        """基于belief机制生成对抗者ID列表"""
        # 获取当前环境观察
        observations = self.env.joint_observation([])
        
        # 使用belief网络计算每个智能体的对抗倾向
        belief_probs, self.belief_rnn_states = self.belief_net(
            observations,
            self.belief_rnn_states,
            self.masks
        )
        
        # 将belief值转换为numpy数组并创建智能体-belief对
        belief_probs = belief_probs.detach().numpy()
        agent_beliefs = list(enumerate(belief_probs))
        
        # 根据belief值排序
        agent_beliefs.sort(key=lambda x: x[1], reverse=True)
        
        # 根据belief阈值选择对抗者
        self.adversary_ids = []
        for agent_id, belief in agent_beliefs:
            if belief > self.belief_threshold:
                self.adversary_ids.append(agent_id)
        
        # 确保对抗者数量在合理范围内
        if is_adversary:
            # 对抗模式下确保至少有min_adversaries个对抗者
            while len(self.adversary_ids) < self.min_adversaries:
                # 从剩余智能体中选择belief值最高的
                remaining = [a for a, _ in agent_beliefs if a not in self.adversary_ids]
                if not remaining:
                    break
                self.adversary_ids.append(remaining[0])
        
        # 确保不超过最大对抗者数量
        if len(self.adversary_ids) > self.max_adversaries:
            # 保留belief值最高的max_adversaries个智能体
            self.adversary_ids = self.adversary_ids[:self.max_adversaries]
        
        # 更新masks用于下一次belief计算
        self.masks = torch.ones(self.nr_agents, 1)
        for agent_id in self.adversary_ids:
            self.masks[agent_id] = 0
        
        return self.adversary_ids

    def update_belief_states(self, observations, rewards):
        """根据观察和奖励更新belief状态"""
        # 获取新的belief值
        with torch.no_grad():
            belief_probs, new_belief_rnn_states = self.belief_net(
                observations,
                self.belief_rnn_states,
                self.masks
            )
        
        # 更新RNN状态
        self.belief_rnn_states = new_belief_rnn_states
        
        return belief_probs

    def get_nr_adversaries(self):
        """获取当前对抗者数量"""
        return len(self.adversary_ids)

    def get_nr_protagonists(self):
        """获取当前主角数量"""
        return self.nr_agents - self.get_nr_adversaries()

    def policy(self, observations, training_mode=True):
        # 根据观察值生成随机联合动作
        random_joint_action = [random.choice(self.actions) \
            for agent in self.env.agents]
        return random_joint_action

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, dones, is_adversary):
        # 更新函数，当前未实现具体逻辑
        return True

class DeepLearningController(Controller):
    
    def __init__(self, params):
        super(DeepLearningController, self).__init__(params)
        self.device = torch.device("cpu")
        self.use_global_reward = get_param_or_default(params, "use_global_reward", True)
        self.input_shape = params["local_observation_shape"]
        self.global_input_shape = params["global_observation_shape"]
        self.memory = ReplayMemory(params["memory_capacity"])
        self.warmup_phase = params["warmup_phase"]
        self.episode_transitions = []                
        self.max_history_length = get_param_or_default(params, "max_history_length", 1)
        self.target_update_period = params["target_update_period"]
        self.epsilon = 1
        self.training_count = 0
        self.current_histories = []
        self.current_pro_histories = []
        self.current_adv_histories = []
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.policy_net = None
        self.target_net = None
        self.default_observations = [numpy.zeros(self.input_shape) for _ in range(self.nr_agents)]

    def save_weights(self, path):
        # 保存策略网络的权重
        if self.policy_net is not None:
            self.policy_net.save_weights(path)

    def load_weights(self, path):
        # 加载策略网络的权重
        if self.policy_net is not None:
            self.policy_net.load_weights(path)

    def extend_histories(self, histories, observations):
        # 扩展历史记录
        histories = histories + [observations]
        return pad_or_truncate_sequences(histories, self.max_history_length, self.default_observations)

    def policy(self, observations, training_mode=True):
        # 根据观察值生成联合动作概率
        self.current_histories = self.extend_histories(self.current_histories, observations)
        action_probs = self.joint_action_probs(self.current_histories, training_mode)
        return [numpy.random.choice(self.actions, p=probs) for probs in action_probs]

    def joint_action_probs(self, histories, training_mode=True, agent_ids=None):
        # 生成联合动作概率
        if agent_ids is None:
            agent_ids = self.agent_ids
        return [numpy.ones(self.nr_actions)/self.nr_actions for _ in agent_ids]

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, dones, is_adversary):
        # 更新函数，调用更新转换函数
        return self.update_transition(state, observations, joint_action, rewards, next_state, next_observations, dones, is_adversary)

    def update_transition(self, state, obs, joint_action, rewards, next_state, next_obs, dones, is_adversary):
        # 更新转换函数
        self.warmup_phase = max(0, self.warmup_phase-1)  # 减少预热阶段计数，确保不低于0
        if self.use_global_reward:  # 如果使用全局奖励
            global_reward = sum(rewards)  # 计算全局奖励为所有奖励的和
            rewards = [global_reward for _ in range(self.nr_agents)]  # 将全局奖励分配给每个代理
        pro_obs = []  # 初始化主角观察列表
        adv_obs = []  # 初始化对手观察列表
        pro_actions = []  # 初始化主角动作列表
        adv_actions = []  # 初始化对手动作列表
        next_pro_obs = []  # 初始化下一步主角观察列表
        next_adv_obs = []  # 初始化下一步对手观察列表
        pro_rewards = []  # 初始化主角奖励列表
        adv_rewards = []  # 始化对手奖励列表
        for i in range(self.nr_agents):  # 遍历每个代理
            if i in self.adversary_ids:  # 如果代理是对手
                adv_obs.append(obs[i])  # 添加对手观察
                adv_actions.append(joint_action[i])  # 添加对手动作
                next_adv_obs.append(next_obs[i])  # 添加下一步对手观察
                adv_rewards.append(-rewards[i])  # 添加对手奖励（取反）
            else:  # 如果代理是主角
                pro_obs.append(obs[i])  # 添加主角观察
                pro_actions.append(joint_action[i])  # 添加主角动作
                next_pro_obs.append(next_obs[i])  # 添加下一步主角观察
                pro_rewards.append(rewards[i])  # 添加主角奖励
        protagonist_ids = [i for i in self.agent_ids if i not in self.adversary_ids]  # 获取所有主角的ID
        self.current_pro_histories = self.extend_histories(self.current_pro_histories, pro_obs)  # 扩展主角历史记录
        self.current_adv_histories = self.extend_histories(self.current_adv_histories, adv_obs)  # 扩展对手历史记录
        pro_probs = self.joint_action_probs(self.current_pro_histories, training_mode=True, agent_ids=protagonist_ids)  # 计算主角联合动作概率
        adv_probs = self.joint_action_probs(self.current_adv_histories, training_mode=True, agent_ids=self.adversary_ids)  # 计算对手联合动作概率
        self.episode_transitions.append((state, pro_obs, adv_obs, pro_actions, adv_actions,\
            pro_probs, adv_probs,pro_rewards, adv_rewards, next_state, next_pro_obs, next_adv_obs))  # 添加当前转换到episode_transitions
        global_terminal_reached = not [d for i,d in enumerate(dones) if (not d) and (i not in self.adversary_ids)]  # 检查是否达到全局终止状态
        if global_terminal_reached:  # 如果达到全局终止状态
            s, pro_obs, adv_obs, a1, a2, p1, p2, pro_rewards, adv_rewards, sn, next_pro_obs, next_adv_obs = tuple(zip(*self.episode_transitions))  # 解压缩episode_transitions
            R1 = self.to_returns(pro_rewards, protagonist_ids)  # 计算主角的回报
            R2 = self.to_returns(adv_rewards, self.adversary_ids)  # 计算对手的回报
            self.memory.save((s, pro_obs, adv_obs, a1, a2, p1, p2, pro_rewards, adv_rewards, sn, next_pro_obs, next_adv_obs, R1,R2))  # 保存转换到记忆
            self.episode_transitions.clear()  # 清空episode_transitions
            self.current_histories.clear()  # 清空当前历史记录
            self.current_pro_histories.clear()  # 清空当前主角历史记录
            self.current_adv_histories.clear()  # 清空当前对手历史记���
        return True  # 返回True表示更新成功

    def collect_minibatch_data(self, minibatch, whole_batch=False):
        # 收集小批量数据
        states = []  # 状态列表
        pro_histories = []  # 主角历史记录列表
        adv_histories = []  # 对手历史记录列表
        next_states = []  # 下一步状态列表
        next_pro_histories = []  # 下一步主角历史记录列表
        next_adv_histories = []  # 下一步对手历史记录列表
        pro_returns = []  # 主角回报列表
        adv_returns = []  # 对手回报列表
        pro_action_probs = []  # 主角动作概率列表
        adv_action_probs = []  # 对手动作概率列表
        pro_actions = []  # 主角动作列表
        adv_actions = []  # 对手动作列表
        pro_rewards = []  # 主角奖励列表
        adv_rewards = []  # 对手奖励列表
        max_length = self.max_history_length  # 最大历史记录长度
        for episode in minibatch:  # 遍历小批量中的每个episode
            states_, pro_obs, adv_obs, p_actions, a_actions, pro_probs, adv_probs,\
                p_rewards, a_rewards, next_states_, next_pro_obs, next_adv_obs, p_R, a_R = episode  # 解包episode数据
            min_index = -max_length+1  # 最小索引
            max_index = len(pro_obs)-max_length  # 最大索引
            if whole_batch:  # 如果处理整个批量
                indices = range(min_index, max_index)  # 使用整个范围的索引
            else:
                indices = [numpy.random.randint(min_index, max_index)]  # 随机选择一个索引
            for index_ in indices:  # 遍历选择的索引
                end_index = index_+max_length  # 计算结束索引
                index = max(0, index_)  # 确保索引不小于0
                assert index < end_index  # 确保索引有效
                # 填充或截断主角观察序列
                history = pad_or_truncate_sequences(list(pro_obs[index:index+max_length]), max_length, self.default_observations)
                pro_histories.append(history)  # 添加到主角历史记录
                # 填充或截断下一步主角观察序列
                next_history = pad_or_truncate_sequences(list(next_pro_obs[index:index+max_length]), max_length, self.default_observations)
                next_pro_histories.append(next_history)  # 添加到下一步主角历史记录
                # 填充或截断对手观察序列
                history = pad_or_truncate_sequences(list(adv_obs[index:index+max_length]), max_length, self.default_observations)
                adv_histories.append(history)  # 添加到对手历史记录
                # 填充或截断下一步对手观察序列
                next_history = pad_or_truncate_sequences(list(next_adv_obs[index:index+max_length]), max_length, self.default_observations)
                next_adv_histories.append(next_history)  # 添加到下一步对手历史记录
                states.append(states_[end_index-1])  # 添加状态
                next_states.append(next_states_[end_index-1])  # 添加下一步状态
                pro_action_probs += list(pro_probs[end_index-1])  # 添加主角动作概率
                adv_action_probs += list(adv_probs[end_index-1])  # 添加对手动作概率
                pro_actions += list(p_actions[end_index-1])  # 添加主角动作
                adv_actions += list(a_actions[end_index-1])  # 添加对手动作
                pro_rewards += list(p_rewards[end_index-1])  # 添加主角奖励
                pro_returns += list(p_R[end_index-1])  # 添加主角回报
                adv_rewards += list(a_rewards[end_index-1])  # 添加对手奖励
                adv_returns += list(a_R[end_index-1])  # 添加对手回报
        # 重塑历史记录
        pro_histories = self.reshape_histories(pro_histories)
        next_pro_histories = self.reshape_histories(next_pro_histories)
        adv_histories = self.reshape_histories(adv_histories)
        next_adv_histories = self.reshape_histories(next_adv_histories)
        # 归一化回报
        pro_returns = self.normalized_returns(numpy.array(pro_returns))
        adv_returns = self.normalized_returns(numpy.array(adv_returns))
        # 返回收集的数据
        return {"states":torch.tensor(states, device=self.device, dtype=torch.float32),\
            "pro_histories":torch.tensor(pro_histories, device=self.device, dtype=torch.float32),\
            "adv_histories":torch.tensor(adv_histories, device=self.device, dtype=torch.float32),\
            "pro_actions":torch.tensor(pro_actions, device=self.device, dtype=torch.long),\
            "adv_actions":torch.tensor(adv_actions, device=self.device, dtype=torch.long),\
            "pro_action_probs":torch.tensor(pro_action_probs, device=self.device, dtype=torch.float32),\
            "adv_action_probs":torch.tensor(adv_action_probs, device=self.device, dtype=torch.float32),\
            "pro_rewards":torch.tensor(pro_rewards, device=self.device, dtype=torch.float32),\
            "adv_rewards":torch.tensor(adv_rewards, device=self.device, dtype=torch.float32),\
            "next_states":torch.tensor(next_states, device=self.device, dtype=torch.float32),\
            "next_pro_histories":torch.tensor(next_pro_histories, device=self.device, dtype=torch.float32),\
            "next_adv_histories":torch.tensor(next_adv_histories, device=self.device, dtype=torch.float32),\
            "pro_returns":torch.tensor(pro_returns, device=self.device, dtype=torch.float32),\
            "adv_returns":torch.tensor(adv_returns, device=self.device, dtype=torch.float32)}

    def reshape_histories(self, history_batch):
        # 重塑历史记录
        histories = []
        for i in range(self.max_history_length):
            joint_observations = []
            for joint_history in history_batch:
                joint_observations += joint_history[i]
            histories.append(joint_observations)
        return histories

    def to_returns(self, individual_rewards, agent_ids):
        # 计算折扣回报
        R = numpy.zeros(len(agent_ids))
        discounted_returns = []
        for rewards in reversed(individual_rewards):
            R = numpy.array(rewards) + self.gamma*R
            discounted_returns.append(R)
        discounted_returns.reverse()
        return numpy.array(discounted_returns)

    def normalized_returns(self, discounted_returns):
        # 归一化折扣回报
        R_mean = numpy.mean(discounted_returns)
        R_std = numpy.std(discounted_returns)
        return (discounted_returns - R_mean)/(R_std + self.eps)

    def update_target_network(self):
        # 更新目标网络
        target_net_available = self.target_net is not None
        if target_net_available and self.training_count % self.target_update_period is 0:
            self.target_net.protagonist_net.load_state_dict(self.policy_net.protagonist_net.state_dict())
            self.target_net.protagonist_net.eval()
            self.target_net.adversary_net.load_state_dict(self.policy_net.adversary_net.state_dict())
            self.target_net.adversary_net.eval()
