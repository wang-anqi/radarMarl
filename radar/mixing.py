import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixNet(nn.Module):
    """QMIX网络实现，包含动态权重生成、多层注意力和防御性机制
    
    特点：
    1. 动态权重生成：使用超网络根据全局状态动态生成权重
    2. 多层次注意力：结合状态层面和信念层面的注意力
    3. 防御性机制：通过信念值自动调整智能体权重
    """
    def __init__(self, args, n_agents, state_shape, mixing_embed_dim=32, hypernet_embed=64):
        super(QMixNet, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_shape
        self.mixing_embed_dim = mixing_embed_dim
        self.hypernet_embed = hypernet_embed
        
        # 超网络结构 - 动态权重生成网络
        # 第一层超网络：生成混合网络第一层的权重和偏置
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, mixing_embed_dim * n_agents),
            nn.Tanh()  # 确保权重在合理范围内
        )
        
        # 第二层超网络：生成混合网络第二层的权重
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, mixing_embed_dim),
            nn.Tanh()  # 确保权重在合理范围内
        )
        
        # 偏置网络
        self.hyper_b1 = nn.Sequential(
            nn.Linear(self.state_dim, mixing_embed_dim),
            nn.ReLU()
        )
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
        # 状态层面的注意力网络
        self.state_attention = nn.Sequential(
            nn.Linear(self.state_dim, mixing_embed_dim),
            nn.Tanh(),
            nn.Linear(mixing_embed_dim, n_agents),
            nn.Softmax(dim=-1)  # 确保注意力权重和为1
        )
        
        # 信念层面的注意力网络
        self.belief_attention = nn.Sequential(
            nn.Linear(n_agents, mixing_embed_dim),
            nn.Tanh(),
            nn.Linear(mixing_embed_dim, n_agents),
            nn.Softmax(dim=-1)  # 确保注意力权重和为1
        )
        
        # 注意力融合网络
        self.attention_fusion = nn.Sequential(
            nn.Linear(2 * n_agents, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, n_agents),
            nn.Sigmoid()  # 控制融合权重在[0,1]范围内
        )
        
        # 手动设置恶意智能体的掩码
        self.malicious_mask = torch.ones(n_agents)

    def set_malicious_agents(self, agent_indices):
        """手动设置恶意智能体
        
        参数:
            agent_indices: 恶意智能体的索引列表
        """
        self.malicious_mask = torch.ones(self.n_agents)
        for idx in agent_indices:
            self.malicious_mask[idx] = -1.0  # 恶意智能体的贡献将被反转

    def forward(self, agent_qs, states, beliefs):
        """
        前向传播过程，并打印每个智能体的贡献
        
        参数:
            agent_qs: 各智能体的Q值 [batch_size, n_agents]
            states: 全局状态 [batch_size, state_dim]
            beliefs: 智能体类型的信念值 [batch_size, n_agents]
            
        返回:
            q_total: 混合后的总Q值 [batch_size, 1]
        """
        # 确保输入维度正确
        if len(states.shape) == 1:
            states = states.unsqueeze(0)  # [1, state_dim]
        if len(beliefs.shape) == 1:
            beliefs = beliefs.unsqueeze(0)  # [1, n_agents]
        if len(agent_qs.shape) == 1:
            agent_qs = agent_qs.unsqueeze(0)  # [1, n_agents]
            
        # 获取最大的batch_size
        batch_sizes = [t.size(0) for t in [agent_qs, states, beliefs]]
        batch_size = max(batch_sizes)
        
        # 扩展所有输入到相同的batch_size
        if agent_qs.size(0) == 1 and batch_size > 1:
            agent_qs = agent_qs.expand(batch_size, -1)
        if states.size(0) == 1 and batch_size > 1:
            states = states.expand(batch_size, -1)
        if beliefs.size(0) == 1 and batch_size > 1:
            beliefs = beliefs.expand(batch_size, -1)
        
        # 1. 多层次注意力机制
        # 1.1 状态层面的注意力权重
        state_weights = self.state_attention(states)  # [batch_size, n_agents]
        
        # 1.2 信念层面的注意力权重
        belief_weights = self.belief_attention(beliefs)  # [batch_size, n_agents]
        
        # 1.3 融合两个层面的注意力
        attention_concat = torch.cat([state_weights, belief_weights], dim=-1)  # [batch_size, 2*n_agents]
        attention_weights = self.attention_fusion(attention_concat)  # [batch_size, n_agents]
        
        # 2. 应用恶意智能体掩码
        malicious_mask = self.malicious_mask.to(states.device)
        defensive_weights = 1.0 - beliefs  # [batch_size, n_agents]
        
        # 3. 最终权重：结合注意力权重、防御权重和恶意掩码
        final_weights = attention_weights * defensive_weights * malicious_mask  # [batch_size, n_agents]
        
        # 4. 动态权重生成（通过超网络）
        # 4.1 生成第一层的权重和偏置
        w1 = self.hyper_w1(states)  # [batch_size, mixing_embed_dim * n_agents]
        b1 = self.hyper_b1(states)  # [batch_size, mixing_embed_dim]
        
        # 重塑权重和偏置
        w1 = w1.view(batch_size, self.n_agents, self.mixing_embed_dim)  # [batch_size, n_agents, mixing_embed_dim]
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)  # [batch_size, 1, mixing_embed_dim]
        
        # 4.2 应用权重到智能体Q值
        weighted_qs = agent_qs.unsqueeze(1)  # [batch_size, 1, n_agents]
        weighted_qs = weighted_qs * final_weights.unsqueeze(1)  # [batch_size, 1, n_agents]
        
        # 4.3 第一层混合
        hidden = F.elu(torch.bmm(weighted_qs, w1) + b1)  # [batch_size, 1, mixing_embed_dim]
        
        # 4.4 生成第二层的权重和偏置
        w2 = self.hyper_w2(states)  # [batch_size, mixing_embed_dim]
        b2 = self.hyper_b2(states)  # [batch_size, 1]
        
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)  # [batch_size, mixing_embed_dim, 1]
        b2 = b2.view(batch_size, 1, 1)  # [batch_size, 1, 1]
        
        # 4.5 第二层混合得到最终Q值
        q_total = torch.bmm(hidden, w2) + b2  # [batch_size, 1, 1]
        q_total = q_total.view(batch_size, -1)  # [batch_size, 1]
        
        # 打印每个智能体的贡献占比
        with torch.no_grad():
            contributions = (weighted_qs.squeeze() * w1.mean(dim=2)).sum(dim=1)
            total_contribution = contributions.abs().sum()
            if total_contribution > 0:
                relative_contributions = contributions / total_contribution
                
                print("\n智能体贡献占比:")
                for i in range(self.n_agents):
                    contribution = relative_contributions[i].item() * 100
                    agent_type = "恶意" if self.malicious_mask[i] < 0 else "正常"
                    print(f"智能体 {i} ({agent_type}): {contribution:.2f}%")
                print("-" * 40)
        
        return q_total

class BeliefWeightedVDN(nn.Module):
    """基于信念的VDN实现（用于对比实验）"""
    def __init__(self, n_agents):
        super(BeliefWeightedVDN, self).__init__()
        self.n_agents = n_agents
        
    def forward(self, agent_qs, beliefs):
        """
        使用信念值对智能体Q值进行加权求和
        
        参数:
            agent_qs: 智能体Q值 [batch_size, n_agents]
            beliefs: 信念值 [batch_size, n_agents]
        """
        defensive_weights = 1.0 - beliefs  # 防御性权重
        weighted_qs = agent_qs * defensive_weights  # 加权Q值
        return weighted_qs.sum(dim=-1)  # 求和得到总Q值