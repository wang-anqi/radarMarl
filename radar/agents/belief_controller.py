import torch
from radar.agents.controller import Controller
from radar.belief.belief import Belief

class BeliefController(Controller):
    """基于信念机制的控制器"""
    
    def __init__(self, params):
        super().__init__(params)
        
        # 初始化belief网络及其状态
        self.belief_net = Belief(
            params, 
            params["local_observation_shape"],
            params["nr_actions"],
            params["nr_agents"],
            device=torch.device("cpu")
        )
        
        # 初始化RNN状态
        self.belief_rnn_states = torch.zeros(
            params["nr_agents"],  # batch_size
            params["hidden_sizes"][-1],  # hidden_size
            dtype=torch.float
        )
        
        # 初始化masks
        self.masks = torch.ones(params["nr_agents"], 1, dtype=torch.float)
        
        # 初始化对抗者ID列表
        self.adversary_ids = []
        
        # 设置belief相关参数
        self.belief_threshold = params["belief_threshold"]
        self.min_adversaries = params["min_adversaries"]
        self.max_adversaries = params["max_adversaries"]
        
        # 初始化对抗者比例
        self._adversary_ratio = 0.0
        
        print("\n" + "="*50)
        print("BeliefController Initialized:")
        print("-"*50)
        print(f"Number of agents: {self.nr_agents}")
        print(f"Current adversaries: {len(self.adversary_ids)}")
        print(f"Initial adversary ratio: {self._adversary_ratio}")
        print("="*50 + "\n")
    
    @property
    def adversary_ratio(self):
        """获取当前对抗者比例"""
        return self._adversary_ratio
    
    @adversary_ratio.setter
    def adversary_ratio(self, value):
        """设置对抗者比例"""
        print("\n" + "-"*50)
        print(f"Setting Adversary Ratio: {self._adversary_ratio:.4f} -> {value:.4f}")
        
        self._adversary_ratio = value
        target_num = int(self.nr_agents * value)
        
        # 获取当前环境观察
        observations = self.env.joint_observation([])
        
        # 使用belief网络计算每个智能体的对抗倾向
        with torch.no_grad():
            belief_probs, _ = self.belief_net(
                observations,
                self.belief_rnn_states,
                self.masks
            )
        
        # 将belief值转换为numpy数组
        belief_probs = belief_probs.detach().numpy()
        
        # 获取每个智能体的平均belief值
        agent_beliefs = [(i, belief_probs[i].mean()) for i in range(self.nr_agents)]
        
        # 根据belief值排序
        agent_beliefs.sort(key=lambda x: x[1], reverse=True)
        
        # 选择belief值最高的前target_num个智能体作为对抗者
        self.adversary_ids = [agent_id for agent_id, _ in agent_beliefs[:target_num]]
        
        # 更新masks
        self.masks = torch.ones(self.nr_agents, 1)
        for agent_id in self.adversary_ids:
            self.masks[agent_id] = 0
        
        print(f"Target number of adversaries: {target_num}")
        print(f"Selected adversaries: {self.adversary_ids}")
        print("-"*50 + "\n")
    
    def update_adversary_ratio(self):
        """更新对抗者比例"""
        old_ratio = self._adversary_ratio
        self._adversary_ratio = len(self.adversary_ids) / self.nr_agents if self.nr_agents > 0 else 0.0
        
        if old_ratio != self._adversary_ratio:
            print(f"\nAdversary Ratio Updated: {old_ratio:.4f} -> {self._adversary_ratio:.4f}")
        
        return self._adversary_ratio
    
    def generate_adversary_ids(self, is_adversary):
        """基于belief机制生成对抗者ID列表"""
        print("\n" + "-"*50)
        print("Generating Adversary IDs using Belief Mechanism:")
        print(f"Current adversary ratio: {self._adversary_ratio}")
        print(f"Current adversaries: {self.adversary_ids}")
        
        # 获取当前环境观察
        observations = self.env.joint_observation([])
        
        # 使用belief网络计算每个智能体的对抗倾向
        belief_probs, self.belief_rnn_states = self.belief_net(
            observations,
            self.belief_rnn_states,
            self.masks
        )
        
        # 将belief值转换为numpy数组并计算每个智能体的平均belief值
        belief_probs = belief_probs.detach().numpy()
        agent_beliefs = [(i, float(belief_probs[i].mean())) for i in range(self.nr_agents)]
        
        # 打印每个智能体的belief值
        print("\nAgent Belief Values:")
        for agent_id, belief in agent_beliefs:
            print(f"Agent {agent_id}: {belief:.4f}")
        
        # 根据belief阈值选择对抗者
        old_adversaries = self.adversary_ids.copy()
        self.adversary_ids = []
        for agent_id, belief in agent_beliefs:
            if belief > self.belief_threshold:
                self.adversary_ids.append(agent_id)
        
        # 确保对抗者数量在合理范围内
        if is_adversary:
            while len(self.adversary_ids) < self.min_adversaries:
                remaining = [a for a, _ in agent_beliefs if a not in self.adversary_ids]
                if not remaining:
                    break
                self.adversary_ids.append(remaining[0])
        
        if len(self.adversary_ids) > self.max_adversaries:
            self.adversary_ids = self.adversary_ids[:self.max_adversaries]
        
        # 更新masks和对抗者比例
        self.masks = torch.ones(self.nr_agents, 1)
        for agent_id in self.adversary_ids:
            self.masks[agent_id] = 0
            
        self.update_adversary_ratio()
        
        # 打印变化信息
        print("\nAdversary Changes:")
        print(f"Previous adversaries: {old_adversaries}")
        print(f"New adversaries: {self.adversary_ids}")
        print(f"New adversary ratio: {self._adversary_ratio:.4f}")
        print("-"*50 + "\n")
        
        return self.adversary_ids
    
    def update(self, state, observations, joint_action, rewards, 
              next_state, next_observations, dones, is_adversary):
        """更新控制器状态"""
        # 更新belief状态
        belief_probs = self.update_belief_states(observations, rewards)
        
        # 根据belief更新对抗者列表
        self.generate_adversary_ids(is_adversary)
        
        # 调用父类的更新方法
        return super().update(state, observations, joint_action, rewards,
                            next_state, next_observations, dones, is_adversary)
    
    def policy(self, observations, training_mode=True):
        """根据当前观察生成动作"""
        # 更新belief状态
        belief_probs = self.update_belief_states(observations, None)
        
        # 生成动作
        joint_action = super().policy(observations, training_mode)
        
        return joint_action 
    
    def sample_adversary_ratio(self):
        """
        在信念机制中，不需要随机采样对抗者比例
        而是根据belief值动态确定对抗者
        """
        # 获取当前环境观察
        observations = self.env.joint_observation([])
        
        # 使用belief网络计算每个智能体的对抗倾向
        with torch.no_grad():
            belief_probs, _ = self.belief_net(
                observations,
                self.belief_rnn_states,
                self.masks
            )
        
        # 根据belief值确定对抗者
        belief_probs = belief_probs.detach().numpy()
        agent_beliefs = [(i, float(belief_probs[i].mean())) for i in range(self.nr_agents)]
        
        # 根据belief阈值选择对抗者
        self.adversary_ids = [
            agent_id for agent_id, belief in agent_beliefs 
            if belief > self.belief_threshold
        ]
        
        # 确保对抗者数量在合理范围内
        if len(self.adversary_ids) < self.min_adversaries:
            agent_beliefs.sort(key=lambda x: x[1], reverse=True)
            remaining = [a for a, _ in agent_beliefs if a not in self.adversary_ids]
            while len(self.adversary_ids) < self.min_adversaries and remaining:
                self.adversary_ids.append(remaining.pop(0))
                
        if len(self.adversary_ids) > self.max_adversaries:
            self.adversary_ids = self.adversary_ids[:self.max_adversaries]
        
        # 更新masks和对抗者比例
        self.masks = torch.ones(self.nr_agents, 1)
        for agent_id in self.adversary_ids:
            self.masks[agent_id] = 0
            
        # 更新并返回对抗者比例
        self.update_adversary_ratio()
        return self._adversary_ratio 
    
    def save_weights(self, path):
        """保存belief网络的权重
        
        Args:
            path: 权重文件保存路径
        """
        # 创建包含所有网络状态的字典
        state_dict = {
            'feature_extractor': self.belief_net.feature_extractor.state_dict(),
            'belief_proj': self.belief_net.belief_proj.state_dict()
        }
        
        # 如果使用RNN，也保存RNN的权重
        if self.belief_net.rnn is not None:
            state_dict['rnn'] = self.belief_net.rnn.state_dict()
        
        # 保存当前的RNN状态和masks
        state_dict['rnn_states'] = self.belief_rnn_states
        state_dict['masks'] = self.masks
        
        # 保存其他重要参数
        state_dict['adversary_ids'] = self.adversary_ids
        state_dict['adversary_ratio'] = self._adversary_ratio
        
        # 保存到文件
        torch.save(state_dict, path)
        print(f"Belief network weights saved to {path}")
    
    def load_weights(self, path):
        """加载belief网络的权重
        
        Args:
            path: 权重文件路径
        """
        # 加载状态字典
        state_dict = torch.load(path)
        
        # 加载网络权重
        self.belief_net.feature_extractor.load_state_dict(state_dict['feature_extractor'])
        self.belief_net.belief_proj.load_state_dict(state_dict['belief_proj'])
        
        # 如果有RNN，加载RNN权重
        if self.belief_net.rnn is not None and 'rnn' in state_dict:
            self.belief_net.rnn.load_state_dict(state_dict['rnn'])
        
        # 加载RNN状态和masks
        self.belief_rnn_states = state_dict['rnn_states']
        self.masks = state_dict['masks']
        
        # 加载其他参数
        self.adversary_ids = state_dict['adversary_ids']
        self._adversary_ratio = state_dict['adversary_ratio']
        
        print(f"Belief network weights loaded from {path}") 