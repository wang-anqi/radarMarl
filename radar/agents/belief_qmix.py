import torch
import torch.nn as nn
import torch.nn.functional as F
from radar.mixing import QMixNet
from radar.belief.belief import Belief
import numpy as np
import random

class BeliefQMIXLearner:
    """基于信念的QMIX学习器
    
    将信念系统与QMIX相结合，通过信念值动态调整智能体的重要性，
    实现对潜在对抗性智能体的防御机制。
    """
    
    def __init__(self, params):
        """
        初始化信念QMIX学习器
        
        参数:
            params: 参数字典，包含所有必要的配置
        """
        self.params = params
        self.n_agents = params["nr_agents"]
        self.state_shape = params["state_shape"]
        self.obs_shape = params["obs_shape"]
        self.action_space = params["action_space"]
        self.n_actions = self.action_space.n  # 获取动作空间的大小
        self.device = params.get("device", torch.device("cpu"))
        self.adversary_ratio = 0.0  # 初始化对抗者比例为0
        
        # 初始化对抗者相关参数
        self.adversary_ids = []  # 对抗者ID列表
        self.belief_threshold = params.get("belief_threshold", 0.6)  # 信念阈值
        self.min_adversaries = params.get("min_adversaries", 1)  # 最小对抗者数量
        self.max_adversaries = params.get("max_adversaries", self.n_agents - 1)  # 最大对抗者数量
        
        # 初始化RNN状态和masks，添加更大的随机性
        hidden_size = params.get("hidden_sizes", [64, 64])[-1]
        self.belief_rnn_states = torch.randn(
            self.n_agents,  # batch_size
            hidden_size,  # hidden_size
            dtype=torch.float
        ).to(self.device) * 0.5  # 增加初始值的方差
        
        # 随机初始化masks，使用更大的随机性
        self.masks = torch.bernoulli(torch.rand(self.n_agents, 1) * 0.8 + 0.1).to(self.device)
        
        # 初始化QMIX网络
        self.qmix_net = QMixNet(
            args=params,
            n_agents=self.n_agents,
            state_shape=np.prod(self.state_shape),
            mixing_embed_dim=params.get("mixing_embed_dim", 32),
            hypernet_embed=params.get("hypernet_embed", 64)
        ).to(self.device)
        
        # 初始化目标网络
        self.target_qmix_net = QMixNet(
            args=params,
            n_agents=self.n_agents,
            state_shape=np.prod(self.state_shape),
            mixing_embed_dim=params.get("mixing_embed_dim", 32),
            hypernet_embed=params.get("hypernet_embed", 64)
        ).to(self.device)
        
        # 初始化信念网络，增加输入维度
        self.belief_net = Belief(
            args=params,
            obs_space=self.obs_shape,
            action_space=self.action_space,
            num_agents=self.n_agents,
            device=self.device
        ).to(self.device)
        
        # 复制参数到目标网络
        self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
        
        # 初始化优化器，使用更大的学习率
        self.qmix_optimizer = torch.optim.Adam(
            self.qmix_net.parameters(),
            lr=params.get("learning_rate", 0.001)
        )
        
        self.belief_optimizer = torch.optim.Adam(
            self.belief_net.parameters(),
            lr=params.get("belief_learning_rate", 0.005)  # 增加信念网络的学习率
        )
        
        # 经验回放缓冲区设置
        self.memory_capacity = params.get("memory_capacity", 20000)
        self.batch_size = params.get("batch_size", 32)
        self.memory = []
        
        # 训练相关参数
        self.gamma = params.get("gamma", 0.99)  # 折扣因子
        self.target_update_interval = params.get("target_update_interval", 200)
        self.train_steps = 0
        
        # 信念系统相关参数
        self.belief_update_freq = params.get("belief_update_freq", 10)
        
        # 恶意智能体设置
        self.malicious_agents = []
        
        print("\n初始化信念QMIX学习器:")
        print(f"智能体数量: {self.n_agents}")
        print(f"信念阈值: {self.belief_threshold}")
        print(f"目标网络更新间隔: {self.target_update_interval}")
        print(f"设备: {self.device}\n")
    
    def update_beliefs(self, observations, rnn_states, masks):
        """
        更新智能体类型的信念值
        
        参数:
            observations: 观察值 [batch_size, n_agents, obs_dim]
            rnn_states: RNN隐藏状态
            masks: 智能体掩码
            
        返回:
            beliefs: 更新后的信念值
            new_rnn_states: 新的RNN隐藏状态
        """
        beliefs, new_rnn_states = self.belief_net(observations, rnn_states, masks)
        return beliefs, new_rnn_states
    
    def get_q_values(self, agent_qs, states, beliefs):
        """
        使用信念加权的QMIX获取总Q值
        
        参数:
            agent_qs: 各智能体Q值 [batch_size, n_agents]
            states: 全局状态 [batch_size, state_dim]
            beliefs: 信念值 [batch_size, n_agents]
            
        返回:
            q_total: 混合后的总Q值
        """
        return self.qmix_net(agent_qs, states, beliefs)
    
    def train(self, batch):
        """
        训练QMIX和信念网络
        
        参数:
            batch: 训练数据批次
            
        返回:
            训练损失字典
        """
        # 解包批次数据
        obs_batch = torch.FloatTensor(batch.observations).to(self.device)
        state_batch = torch.FloatTensor(batch.states).to(self.device)
        action_batch = torch.LongTensor(batch.actions).to(self.device)
        reward_batch = torch.FloatTensor(batch.rewards).to(self.device)
        next_obs_batch = torch.FloatTensor(batch.next_observations).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_states).to(self.device)
        done_batch = torch.FloatTensor(batch.dones).to(self.device)
        rnn_states_batch = torch.FloatTensor(batch.rnn_states).to(self.device)
        masks_batch = torch.FloatTensor(batch.masks).to(self.device)
        
        # 获取当前信念值
        beliefs, new_rnn_states = self.belief_net(
            obs_batch, rnn_states_batch, masks_batch
        )
        next_beliefs, _ = self.belief_net(
            next_obs_batch, new_rnn_states, masks_batch
        )
        
        # 计算当前Q值
        current_q_total = self.get_q_values(
            self.get_agent_qvals(obs_batch),
            state_batch,
            beliefs
        )
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_total = self.target_qmix_net(
                self.get_agent_qvals(next_obs_batch),
                next_state_batch,
                next_beliefs
            )
            target_q_total = reward_batch + self.gamma * (1 - done_batch) * next_q_total
        
        # QMIX损失
        qmix_loss = F.mse_loss(current_q_total, target_q_total)
        
        # 信念损失（包含多个组件）
        # 1. 基础信念损失
        belief_base_loss = F.binary_cross_entropy(beliefs, torch.zeros_like(beliefs))
        
        # 2. 信念一致性损失（确保信念变化平滑）
        belief_consistency_loss = F.mse_loss(beliefs, next_beliefs)
        
        # 3. 信念稀疏性损失（鼓励更确定的信念）
        belief_sparsity_loss = torch.mean(torch.abs(beliefs - 0.5))
        
        # 组合信念损失
        belief_loss = (belief_base_loss + 
                      0.1 * belief_consistency_loss + 
                      0.1 * belief_sparsity_loss)
        
        # 更新网络
        # 1. 更新QMIX网络
        self.qmix_optimizer.zero_grad()
        qmix_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qmix_net.parameters(), 1.0)  # 梯度裁剪
        self.qmix_optimizer.step()
        
        # 2. 更新信念网络
        self.belief_optimizer.zero_grad()
        belief_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.belief_net.parameters(), 1.0)  # 梯度裁剪
        self.belief_optimizer.step()
        
        # 定期更新目标网络
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
        
        return {
            'qmix_loss': qmix_loss.item(),
            'belief_loss': belief_loss.item(),
            'belief_base_loss': belief_base_loss.item(),
            'belief_consistency_loss': belief_consistency_loss.item(),
            'belief_sparsity_loss': belief_sparsity_loss.item()
        }
    
    def save(self, path):
        """保存模型检查点"""
        torch.save({
            'qmix_state_dict': self.qmix_net.state_dict(),
            'belief_state_dict': self.belief_net.state_dict(),
            'qmix_optimizer_state_dict': self.qmix_optimizer.state_dict(),
            'belief_optimizer_state_dict': self.belief_optimizer.state_dict(),
            'train_steps': self.train_steps
        }, path)
    
    def load(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.qmix_net.load_state_dict(checkpoint['qmix_state_dict'])
        self.belief_net.load_state_dict(checkpoint['belief_state_dict'])
        self.qmix_optimizer.load_state_dict(checkpoint['qmix_optimizer_state_dict'])
        self.belief_optimizer.load_state_dict(checkpoint['belief_optimizer_state_dict'])
        self.train_steps = checkpoint.get('train_steps', 0)
        self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())

    def generate_adversary_ids(self, is_adversary):
        """基于belief机制生成对抗者ID列表"""
        print("\n" + "-"*50)
        print("Generating Adversary IDs using Belief Mechanism:")
        print(f"Current adversary ratio: {self.adversary_ratio}")
        print(f"Current adversaries: {self.adversary_ids}")
        
        # 获取当前环境观察
        observations = self.params["env"].joint_observation([])
        
        # 使用belief网络计算每个智能体的对抗倾向
        belief_probs, self.belief_rnn_states = self.belief_net(
            observations,
            self.belief_rnn_states,
            self.masks
        )
        
        # 将belief值转换为numpy数组并计算每个智能体的平均belief值
        belief_probs = belief_probs.detach().numpy()
        agent_beliefs = [(i, float(belief_probs[i].mean())) for i in range(self.n_agents)]
        
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
        self.masks = torch.ones(self.n_agents, 1).to(self.device)
        for agent_id in self.adversary_ids:
            self.masks[agent_id] = 0
            
        # 更新对抗者比例
        self.adversary_ratio = len(self.adversary_ids) / self.n_agents if self.n_agents > 0 else 0.0
        
        # 打印变化信息
        print("\nAdversary Changes:")
        print(f"Previous adversaries: {old_adversaries}")
        print(f"New adversaries: {self.adversary_ids}")
        print(f"New adversary ratio: {self.adversary_ratio:.4f}")
        print("-"*50 + "\n")
        
        return self.adversary_ids
        
    def sample_adversary_ratio(self):
        """在信念机制中，根据belief值动态确定对抗者"""
        # 获取当前环境观察
        observations = self.params["env"].joint_observation([])
        
        # 使用belief网络计算每个智能体的对抗倾向
        with torch.no_grad():
            belief_probs, _ = self.belief_net(
                observations,
                self.belief_rnn_states,
                self.masks
            )
        
        # 根据belief值确定对抗者
        belief_probs = belief_probs.detach().numpy()
        agent_beliefs = [(i, float(belief_probs[i].mean())) for i in range(self.n_agents)]
        
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
        self.masks = torch.ones(self.n_agents, 1).to(self.device)
        for agent_id in self.adversary_ids:
            self.masks[agent_id] = 0
            
        # 更新并返回对抗者比例
        self.adversary_ratio = len(self.adversary_ids) / self.n_agents if self.n_agents > 0 else 0.0
        return self.adversary_ratio

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

    def get_agent_qvals(self, observations):
        """获取每个智能体的Q值"""
        # 将观察转换为张量
        obs_tensor = torch.FloatTensor(observations).to(self.device)  # [n_agents, obs_dim]
        
        # 创建一个批次的观察
        batch_obs = obs_tensor.unsqueeze(0)  # [1, n_agents, obs_dim]
        
        # 创建对应的状态和信念
        batch_state = torch.zeros(1, np.prod(self.state_shape)).to(self.device)  # [1, state_dim]
        batch_belief = torch.zeros(1, self.n_agents).to(self.device)  # [1, n_agents]
        
        # 使用QMIX网络计算Q值
        with torch.no_grad():
            # 为每个动作计算Q值
            q_values = torch.zeros(self.n_agents, self.n_actions).to(self.device)
            
            # 对每个动作进行评估
            for action in range(self.n_actions):
                # 创建动作张量
                action_tensor = torch.full((1, self.n_agents), action, dtype=torch.long).to(self.device)
                
                # 创建每个智能体的Q值
                agent_qs = torch.zeros(1, self.n_agents).to(self.device)
                agent_qs[0] = q_values[:, action]  # 使用当前动作的Q值
                
                # 计算联合Q值
                joint_q_value = self.qmix_net.forward(
                    agent_qs,  # [batch_size, n_agents]
                    batch_state,  # [batch_size, state_dim]
                    batch_belief  # [batch_size, n_agents]
                )
                
                # 将Q值分配给对应的动作
                q_values[:, action] = joint_q_value.squeeze() / self.n_agents
        
        return q_values

    def get_agent_qvals_batch(self, observations, actions=None):
        """获取一批智能体的Q值
        
        Args:
            observations: 观察值 [batch_size, n_agents, obs_dim]
            actions: 可选，动作值 [batch_size, n_agents]
            
        Returns:
            q_values: Q值 [batch_size, n_agents, action_space] 或 [batch_size, n_agents] (如果提供了actions)
        """
        # 将观察转换为张量
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        batch_size = obs_tensor.shape[0]
        
        # 创建状态和信念
        batch_state = torch.zeros(batch_size, np.prod(self.state_shape)).to(self.device)
        batch_belief = torch.zeros(batch_size, self.n_agents).to(self.device)
        
        # 使用QMIX网络计算Q值
        with torch.no_grad():
            if actions is not None:
                # 如果提供了动作，直接计算对应的Q值
                action_tensor = torch.LongTensor(actions).to(self.device)
                
                # 创建每个智能体的Q值
                agent_qs = torch.zeros(batch_size, self.n_agents).to(self.device)
                for i in range(batch_size):
                    agent_qs[i] = action_tensor[i]
                
                joint_q_value = self.qmix_net.forward(
                    agent_qs,
                    batch_state,
                    batch_belief
                )
                q_values = joint_q_value.view(batch_size, self.n_agents)
            else:
                # 否则计算所有动作的Q值
                q_values = torch.zeros(batch_size, self.n_agents, self.n_actions).to(self.device)
                for action in range(self.n_actions):
                    # 创建每个智能体的Q值
                    agent_qs = torch.zeros(batch_size, self.n_agents).to(self.device)
                    for i in range(batch_size):
                        agent_qs[i] = q_values[i, :, action]
                    
                    joint_q_value = self.qmix_net.forward(
                        agent_qs,
                        batch_state,
                        batch_belief
                    )
                    q_values[:, :, action] = joint_q_value.view(batch_size, self.n_agents)
        
        return q_values

    def policy(self, observations, training_mode=True):
        """根据当前观察生成动作"""
        # 更新belief状态
        belief_probs = self.update_belief_states(observations, None)
        
        # 获取每个智能体的Q值
        q_values = self.get_agent_qvals(observations)
        
        # 选择每个智能体的动作
        actions = []
        for agent_id in range(self.n_agents):
            if training_mode:
                # 训练模式下使用epsilon-greedy策略
                if random.random() < self.params.get("epsilon", 0.1):
                    action = random.randint(0, self.n_actions - 1)
                else:
                    action = q_values[agent_id].argmax().item()
            else:
                # 测试模式下直接选择最优动作
                action = q_values[agent_id].argmax().item()
            actions.append(action)
        
        return actions

    def update(self, state, observations, joint_action, rewards, 
              next_state, next_observations, dones, is_adversary):
        """更新网络参数"""
        # 更新belief状态
        belief_probs = self.update_belief_states(observations, rewards)
        
        # 如果belief_probs是3维，取平均值转换为2维
        if len(belief_probs.shape) == 3:
            belief_probs = belief_probs.mean(dim=-1)  # [batch_size, n_agents]
        
        # 根据belief更新对抗者列表
        self.generate_adversary_ids(is_adversary)
        
        # 将数据转换为张量并确保维度正确
        state_tensor = torch.FloatTensor(state).view(1, -1).to(self.device)  # [1, state_dim]
        next_state_tensor = torch.FloatTensor(next_state).view(1, -1).to(self.device)  # [1, state_dim]
        obs_tensor = torch.FloatTensor(observations).to(self.device)  # [n_agents, obs_dim]
        next_obs_tensor = torch.FloatTensor(next_observations).to(self.device)  # [n_agents, obs_dim]
        action_tensor = torch.LongTensor(joint_action).to(self.device)  # [n_agents]
        reward_tensor = torch.FloatTensor(rewards).to(self.device)  # [n_agents]
        done_tensor = torch.FloatTensor(dones).to(self.device)  # [n_agents]
        
        # 计算当前Q值
        current_q_values = self.get_agent_qvals(observations)  # [n_agents, n_actions]
        next_q_values = self.get_agent_qvals(next_observations)  # [n_agents, n_actions]
        
        # 计算目标Q值
        with torch.no_grad():
            # 获取每个智能体的最大Q值动作
            next_actions = next_q_values.max(dim=1)[1]  # [n_agents]
            
            # 创建next_agent_qs
            next_agent_qs = torch.zeros(1, self.n_agents).to(self.device)
            for i in range(self.n_agents):
                next_agent_qs[0, i] = next_q_values[i, next_actions[i]]
            
            # 使用目标网络计算目标Q值
            target_q_total = self.target_qmix_net(
                next_agent_qs,  # [1, n_agents]
                next_state_tensor,  # [1, state_dim]
                belief_probs.unsqueeze(0) if len(belief_probs.shape) == 1 else belief_probs  # [1, n_agents]
            )
            
            # 计算TD目标
            td_target = reward_tensor.mean() + (1 - done_tensor.mean()) * self.gamma * target_q_total
        
        # 创建current_agent_qs
        current_agent_qs = torch.zeros(1, self.n_agents).to(self.device)
        for i in range(self.n_agents):
            current_agent_qs[0, i] = current_q_values[i, action_tensor[i]]
        
        # 计算当前Q值
        current_q_total = self.qmix_net(
            current_agent_qs,  # [1, n_agents]
            state_tensor,  # [1, state_dim]
            belief_probs.unsqueeze(0) if len(belief_probs.shape) == 1 else belief_probs  # [1, n_agents]
        )
        
        # 计算TD误差
        td_error = (td_target - current_q_total).detach()
        
        # 计算QMIX损失
        qmix_loss = F.mse_loss(current_q_total, td_target.detach())
        
        # 计算信念损失，使用更复杂的损失函数
        belief_base_loss = F.binary_cross_entropy(belief_probs, torch.zeros_like(belief_probs))
        belief_diversity_loss = -torch.std(belief_probs)  # 鼓励信念值的多样性
        belief_temporal_loss = F.mse_loss(belief_probs, self.masks.squeeze())  # 时间一致性
        
        belief_loss = (belief_base_loss + 
                      0.1 * belief_diversity_loss + 
                      0.1 * belief_temporal_loss)
        
        # 总损失
        total_loss = qmix_loss + 0.2 * belief_loss  # 增加信念损失的权重
        
        # 更新网络
        self.qmix_optimizer.zero_grad()
        self.belief_optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.qmix_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.belief_net.parameters(), 1.0)
        
        # 执行优化步骤
        self.qmix_optimizer.step()
        self.belief_optimizer.step()
        
        # 定期更新目标网络
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
        
        return True

    def set_malicious_agents(self, agent_indices):
        """设置恶意智能体
        
        参数:
            agent_indices: 恶意智能体的索引列表
        """
        self.malicious_agents = agent_indices
        self.qmix_net.set_malicious_agents(agent_indices)
        self.target_qmix_net.set_malicious_agents(agent_indices)
        print(f"\n已设置恶意智能体: {agent_indices}")