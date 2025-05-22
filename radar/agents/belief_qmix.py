import torch
import torch.nn as nn
import torch.nn.functional as F
from radar.mixing import QMixNet
from radar.belief.belief import Belief
import numpy as np

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
        self.device = params.get("device", torch.device("cpu"))
        
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
        
        # 初始化信念网络
        self.belief_net = Belief(
            args=params,
            obs_space=self.obs_shape,
            action_space=self.action_space,
            num_agents=self.n_agents,
            device=self.device
        ).to(self.device)
        
        # 复制参数到目标网络
        self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
        
        # 初始化优化器
        self.qmix_optimizer = torch.optim.Adam(
            self.qmix_net.parameters(),
            lr=params.get("learning_rate", 0.001)
        )
        
        self.belief_optimizer = torch.optim.Adam(
            self.belief_net.parameters(),
            lr=params.get("belief_learning_rate", 0.001)
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
        self.belief_threshold = params.get("belief_threshold", 0.6)
        self.belief_update_freq = params.get("belief_update_freq", 10)
        
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
            self.get_agent_qvals(obs_batch, action_batch),
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