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
        
        # 初始化RNN状态和masks
        hidden_size = params.get("hidden_sizes", [64, 64])[-1]
        self.belief_rnn_states = torch.zeros(
            1,  # batch_size
            self.n_agents,  # number of agents
            hidden_size  # hidden_size
        ).to(self.device)
        
        # 初始化masks，确保维度正确
        self.masks = torch.ones(1, self.n_agents, 1).to(self.device)
        
        # 初始化prev_belief_probs
        self.prev_belief_probs = torch.ones(1, self.n_agents).to(self.device) * 0.5
        
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
        
        # 添加梯度裁剪参数
        self.grad_clip = 10.0
        # 调整学习率
        self.belief_optimizer = torch.optim.Adam(self.belief_net.parameters(), lr=0.001)
        self.qmix_optimizer = torch.optim.Adam(self.qmix_net.parameters(), lr=0.001)
        
        # 添加信念更新计数器
        self.belief_update_counter = 0
        self.min_belief_updates = 1000  # 最小更新次数
        
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
        
        # 打印初始化信息
        print("\n初始化信念QMIX学习器:")
        print(f"智能体数量: {self.n_agents}")
        print(f"信念阈值: {self.belief_threshold}")
        print(f"目标网络更新间隔: {self.target_update_interval}")
        print(f"设备: {self.device}")
        print(f"RNN状态形状: {self.belief_rnn_states.shape}")
        print(f"Masks形状: {self.masks.shape}")
        print(f"初始信念值形状: {self.prev_belief_probs.shape}\n")
    
    def update_belief_states(self, observations, masks):
        """更新信念状态"""
        try:
            # 确保observations是tensor并且维度正确
            if not isinstance(observations, torch.Tensor):
                observations = torch.FloatTensor(observations).to(self.device)
            
            # 添加batch维度如果需要
            if observations.dim() == 2:
                observations = observations.unsqueeze(0)  # [1, n_agents, obs_dim]
            
            # 确保masks维度正确
            if masks is None:
                masks = torch.ones(1, self.n_agents, 1).to(self.device)
            else:
                if not isinstance(masks, torch.Tensor):
                    masks = torch.FloatTensor(masks).to(self.device)
                if masks.dim() == 2:
                    masks = masks.unsqueeze(0)
            
            # 获取当前belief
            with torch.no_grad():
                belief_probs, new_rnn_states = self.belief_net(
                    observations,
                    self.belief_rnn_states,
                    masks
                )
            
            # 确保belief_probs和rnn_states不为None
            if belief_probs is None:
                belief_probs = self.prev_belief_probs
            if new_rnn_states is not None:
                self.belief_rnn_states = new_rnn_states
            
            # 确保belief_probs是正确的形状 [batch_size, n_agents]
            if belief_probs.dim() > 2:
                belief_probs = belief_probs.mean(dim=-1)
            if belief_probs.dim() == 1:
                belief_probs = belief_probs.unsqueeze(0)
            
            # 使用简单的平滑更新
            alpha = 0.8
            belief_probs = alpha * belief_probs + (1 - alpha) * self.prev_belief_probs
            
            # 更新历史信念值
            self.prev_belief_probs = belief_probs.detach()
            
            # 打印调试信息
            if self.train_steps % 100 == 0:
                print("\n信念更新信息:")
                print(f"信念值形状: {belief_probs.shape}")
                print(f"信念值范围: [{belief_probs.min().item():.4f}, {belief_probs.max().item():.4f}]")
                print(f"信念值均值: {belief_probs.mean().item():.4f}")
                print(f"信念值标准差: {belief_probs.std().item():.4f}")
                print("-" * 40)
            
            return belief_probs
            
        except Exception as e:
            print(f"更新信念状态时出错: {str(e)}")
            print(f"错误发生时的状态:")
            print(f"observations shape: {observations.shape if isinstance(observations, torch.Tensor) else 'N/A'}")
            print(f"masks shape: {masks.shape if isinstance(masks, torch.Tensor) else 'N/A'}")
            print(f"belief_rnn_states shape: {self.belief_rnn_states.shape if isinstance(self.belief_rnn_states, torch.Tensor) else 'N/A'}")
            
            # 返回默认信念值
            return self.prev_belief_probs.clone()
    
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
    
    def train(self, batch, t_env, episode):
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
        
        # 计算信念损失
        belief_loss = self.compute_belief_loss(batch)
        
        # 应用梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.belief_net.parameters(), self.grad_clip)
        
        # 更新信念网络
        self.belief_optimizer.zero_grad()
        belief_loss.backward()
        self.belief_optimizer.step()
        
        self.belief_update_counter += 1
        
        # 只有在足够的更新次数后才开始使用信念值
        if self.belief_update_counter < self.min_belief_updates:
            return
            
        # 更新网络
        # 1. 更新QMIX网络
        self.qmix_optimizer.zero_grad()
        qmix_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qmix_net.parameters(), 1.0)  # 梯度裁剪
        self.qmix_optimizer.step()
        
        # 定期更新目标网络
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
        
        return {
            'qmix_loss': qmix_loss.item(),
            'belief_loss': belief_loss.item()
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
        print("基于信念机制识别对抗性智能体:")
        
        # 获取当前环境观察
        observations = self.params["env"].joint_observation([])
        
        # 使用belief网络计算每个智能体的对抗倾向
        belief_probs, self.belief_rnn_states = self.belief_net(
            observations,
            self.belief_rnn_states,
            self.masks
        )
        
        # 将belief值转换为numpy数组并计算每个智能体的平均belief值
        belief_probs = belief_probs.detach().cpu().numpy()
        if len(belief_probs.shape) > 2:  # 如果维度大于2，取平均
            belief_probs = belief_probs.mean(axis=-1)
        
        # 确保belief_probs是二维的 [batch_size, n_agents]
        if len(belief_probs.shape) == 1:
            belief_probs = belief_probs.reshape(1, -1)
            
        # 计算每个智能体的平均belief值
        agent_beliefs = [(i, float(belief_probs[0, i])) for i in range(self.n_agents)]
        
        # 打印每个智能体的信息
        print("\n智能体信念状态:")
        for agent_id, belief in agent_beliefs:
            print(f"智能体 {agent_id}:")
            print(f"  - 信念值: {belief:.4f} (越高表示越可能是对抗性智能体)")
            print(f"  - 当前状态: {'可能是对抗性' if belief > self.belief_threshold else '正常'}")
        
        # 根据belief阈值识别对抗者
        self.adversary_ids = [
            agent_id for agent_id, belief in agent_beliefs 
            if belief > self.belief_threshold
        ]
        
        # 确保对抗者数量在合理范围内
        if len(self.adversary_ids) < self.min_adversaries:
            # 如果检测到的对抗者太少，添加得分最高的智能体
            remaining_agents = sorted(
                [(i, b) for i, b in agent_beliefs if i not in self.adversary_ids],
                key=lambda x: x[1],
                reverse=True
            )
            additional_agents = [
                agent_id for agent_id, _ in remaining_agents[:self.min_adversaries - len(self.adversary_ids)]
            ]
            self.adversary_ids.extend(additional_agents)
        elif len(self.adversary_ids) > self.max_adversaries:
            # 如果检测到的对抗者太多，只保留得分最高的
            agent_scores = sorted(
                [(i, b) for i, b in agent_beliefs if i in self.adversary_ids],
                key=lambda x: x[1],
                reverse=True
            )
            self.adversary_ids = [agent_id for agent_id, _ in agent_scores[:self.max_adversaries]]
        
        print(f"\n信念阈值: {self.belief_threshold:.4f}")
        print(f"识别出的对抗性智能体: {self.adversary_ids}")
        print(f"对抗性智能体数量: {len(self.adversary_ids)}/{self.n_agents}")
        print("-"*50)
        
        return self.adversary_ids
        
    def sample_adversary_ratio(self):
        """基于信念值动态确定对抗者比例"""
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
        
        # 根据belief阈值识别对抗者
        self.adversary_ids = [
            agent_id for agent_id, belief in agent_beliefs 
            if belief > self.belief_threshold
        ]
        
        # 更新masks
        self.masks = torch.ones(self.n_agents, 1).to(self.device)
        for agent_id in self.adversary_ids:
            self.masks[agent_id] = 0
        
        # 计算当前对抗者比例
        self.adversary_ratio = len(self.adversary_ids) / self.n_agents if self.n_agents > 0 else 0.0
        return self.adversary_ratio

    def get_agent_qvals(self, observations):
        """获取每个智能体的Q值"""
        try:
            # 将观察转换为张量
            if not isinstance(observations, torch.Tensor):
                obs_tensor = torch.FloatTensor(observations).to(self.device)  # [n_agents, obs_dim]
            else:
                obs_tensor = observations.to(self.device)
            
            # 创建一个批次的观察
            batch_obs = obs_tensor.unsqueeze(0) if obs_tensor.dim() == 2 else obs_tensor  # [1, n_agents, obs_dim]
            
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
                    try:
                        joint_q_value = self.qmix_net.forward(
                            agent_qs,  # [batch_size, n_agents]
                            batch_state,  # [batch_size, state_dim]
                            batch_belief  # [batch_size, n_agents]
                        )
                        
                        # 将Q值分配给对应的动作
                        if joint_q_value is not None:
                            q_values[:, action] = joint_q_value.squeeze() / max(1, self.n_agents)
                        else:
                            q_values[:, action] = torch.zeros(self.n_agents).to(self.device)
                            
                    except Exception as e:
                        print(f"计算Q值时出错: {str(e)}")
                        q_values[:, action] = torch.zeros(self.n_agents).to(self.device)
            
            return q_values
            
        except Exception as e:
            print(f"获取智能体Q值时出错: {str(e)}")
            # 返回默认Q值
            return torch.zeros(self.n_agents, self.n_actions).to(self.device)

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
        """根据当前观察生成动作
        
        参数:
            observations: 观察值
            training_mode: 是否为训练模式
        返回:
            actions: 动作列表
        """
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
        belief_probs = self.update_belief_states(observations, None)
        
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

    def compute_belief_loss(self, batch):
        """改进的信念损失计算，增加差异化训练信号"""
        try:
            # 获取batch中的观察和奖励
            observations = batch.get('obs', torch.zeros(1, self.n_agents, self.obs_shape))
            rewards = batch.get('reward', torch.zeros(1, self.n_agents))
            actions = batch.get('actions', torch.zeros(1, self.n_agents))
            
            # 确保数据类型正确
            if not isinstance(observations, torch.Tensor):
                observations = torch.FloatTensor(observations).to(self.device)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.FloatTensor(rewards).to(self.device)
            if not isinstance(actions, torch.Tensor):
                actions = torch.FloatTensor(actions).to(self.device)
            
            # 计算信念值
            beliefs, _ = self.belief_net(observations, None, None)
            
            # 确保beliefs不为None且维度正确
            if beliefs is None or beliefs.nelement() == 0:
                beliefs = torch.ones(observations.size(0), self.n_agents).to(self.device) * 0.5
            
            # 1. 基于奖励的自监督损失 - 使用相对奖励
            batch_rewards = rewards.view(-1, self.n_agents)
            reward_mean = batch_rewards.mean(dim=1, keepdim=True)
            reward_std = batch_rewards.std(dim=1, keepdim=True) + 1e-6
            normalized_rewards = (batch_rewards - reward_mean) / reward_std
            reward_based_target = torch.sigmoid(normalized_rewards)
            base_loss = F.binary_cross_entropy(beliefs, reward_based_target)
            
            # 2. 动作差异性损失
            action_diff = torch.abs(actions.unsqueeze(2) - actions.unsqueeze(1))
            action_similarity = 1.0 - action_diff / self.n_actions
            action_based_loss = F.mse_loss(beliefs, action_similarity.mean(dim=-1))
            
            # 3. 信念多样性损失
            belief_mean = beliefs.mean(dim=1, keepdim=True)
            diversity_loss = -torch.mean(torch.abs(beliefs - belief_mean))
            
            # 4. 时间一致性损失
            if beliefs.size(0) > 1:
                consistency_loss = F.mse_loss(beliefs[:-1], beliefs[1:])
            else:
                consistency_loss = torch.tensor(0.0).to(self.device)
            
            # 5. 熵正则化
            entropy = -(beliefs * torch.log(beliefs + 1e-10) + 
                       (1 - beliefs) * torch.log(1 - beliefs + 1e-10)).mean()
            
            # 6. 对抗性检测损失 - 基于观察差异
            obs_diff = torch.cdist(observations.view(observations.size(0), self.n_agents, -1), 
                                 observations.view(observations.size(0), self.n_agents, -1))
            obs_similarity = torch.exp(-obs_diff.mean(dim=-1))
            detection_loss = F.mse_loss(beliefs, 1.0 - obs_similarity)
            
            # 组合所有损失，使用不同的权重
            total_loss = (base_loss * 1.0 +
                         action_based_loss * 0.5 +
                         diversity_loss * 0.3 +
                         consistency_loss * 0.2 +
                         detection_loss * 0.4 -
                         entropy * 0.1)  # 负号是因为我们要最大化熵
            
            # 打印调试信息
            if self.train_steps % 100 == 0:
                print(f"\n信念损失组成:")
                print(f"基础损失: {base_loss.item():.4f}")
                print(f"动作差异损失: {action_based_loss.item():.4f}")
                print(f"多样性损失: {diversity_loss.item():.4f}")
                print(f"一致性损失: {consistency_loss.item():.4f}")
                print(f"检测损失: {detection_loss.item():.4f}")
                print(f"熵: {entropy.item():.4f}")
                print(f"总损失: {total_loss.item():.4f}")
                print("-" * 40)
            
            return total_loss
            
        except Exception as e:
            print(f"计算信念损失时出错: {str(e)}")
            return torch.tensor(0.1).to(self.device)

    def run_episode(self, episode_id, controller, params, is_adversary, training_mode=True, log_level=0, reset_episode=True):
        """运行单个episode"""
        env = params["env"]
        path = params["directory"]
        save_summaries = params["save_summaries"]
        nr_agents = params["nr_agents"]
        
        # 初始化返回值
        protagonist_discounted_return = 0.0
        protagonist_undiscounted_return = 0.0
        policy_updated = False
        time_step = 0
        
        # 生成对抗者ID
        adversary_ids = self.generate_adversary_ids(is_adversary)
        
        # 重置环境
        if reset_episode:
            observations = env.reset(adversary_ids)
        else:
            observations = env.joint_observation(adversary_ids)
        
        state = env.global_state()
        done = False
        
        while not done:
            # 获取动作
            joint_action = self.policy(observations, training_mode)
            
            # 执行动作
            next_observations, rewards, dones, info = env.step(joint_action, adversary_ids)
            next_state = env.global_state()
            
            # 计算奖励
            if len(adversary_ids) < nr_agents:
                nr_protagonists = float(nr_agents - len(adversary_ids))
                protagonist_reward = sum([r/nr_protagonists for i,r in enumerate(rewards) if i not in adversary_ids])
            else:
                protagonist_reward = 0.0
            
            # 更新累积奖励
            protagonist_discounted_return += (params.get("gamma", 0.99) ** time_step) * protagonist_reward
            protagonist_undiscounted_return += protagonist_reward
            
            # 更新策略
            if training_mode:
                policy_updated = self.update(
                    state, 
                    observations, 
                    joint_action, 
                    rewards,
                    next_state, 
                    next_observations, 
                    dones, 
                    is_adversary
                )
            
            # 更新状态
            state = next_state
            observations = next_observations
            time_step += 1
            
            # 检查是否结束
            done = all(dones) or time_step >= params.get("max_episode_steps", 1000)
        
        return float(protagonist_discounted_return), float(protagonist_undiscounted_return), bool(policy_updated), int(time_step)