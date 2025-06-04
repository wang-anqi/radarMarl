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
            nn.Softmax(dim=-1)
        )
        
        # 信念层面的注意力网络
        self.belief_attention = nn.Sequential(
            nn.Linear(n_agents, mixing_embed_dim),
            nn.Tanh(),
            nn.Linear(mixing_embed_dim, n_agents),
            nn.Softmax(dim=-1)
        )
        
        # 注意力融合网络
        self.attention_fusion = nn.Sequential(
            nn.Linear(2 * n_agents, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, n_agents),
            nn.Sigmoid()
        )
        
        # 添加对抗性分析网络
        self.adversary_analyzer = nn.Sequential(
            nn.Linear(mixing_embed_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
            nn.Sigmoid()
        )
        
        # 添加贡献度追踪
        self.contribution_history = {i: [] for i in range(n_agents)}
        self.adversary_score_history = {i: [] for i in range(n_agents)}

    def analyze_agent_contributions(self, weighted_qs, w1, beliefs):
        """分析每个智能体的贡献度和对抗性"""
        try:
            with torch.no_grad():
                # 确保维度正确
                if weighted_qs.dim() == 2:
                    weighted_qs = weighted_qs.unsqueeze(0)
                if w1.dim() == 2:
                    w1 = w1.unsqueeze(0)
                if beliefs.dim() == 1:
                    beliefs = beliefs.unsqueeze(0)
                
                # 计算每个智能体的贡献
                contributions = (weighted_qs.squeeze() * w1.mean(dim=2)).sum(dim=1)
                total_contribution = contributions.abs().sum()
                
                if total_contribution > 0:
                    relative_contributions = contributions / total_contribution
                    
                    # 计算每个智能体的对抗性得分
                    adversary_scores = torch.sigmoid(w1.mean(dim=2)).mean(dim=1)
                    
                    print("\n智能体贡献分析详情:")
                    print("-" * 80)
                    print("智能体ID | 贡献占比 | 信念值 | 对抗性得分 | 权重均值 | Q值均值 | 行为特征")
                    print("-" * 80)
                    
                    for i in range(self.n_agents):
                        if i < relative_contributions.size(0):
                            contribution = relative_contributions[i].item() * 100
                            belief_value = beliefs[0][i].item() if i < beliefs.size(1) else 0.5
                            adversary_score = adversary_scores[i].item() if i < adversary_scores.size(0) else 0.5
                            weight_mean = w1[0, i].mean().item()
                            q_mean = weighted_qs[0, 0, i].item()
                            
                            # 更新历史记录
                            self.contribution_history[i].append(contribution)
                            self.adversary_score_history[i].append(adversary_score)
                            
                            # 计算历史统计
                            if len(self.contribution_history[i]) > 100:
                                self.contribution_history[i].pop(0)
                                self.adversary_score_history[i].pop(0)
                            
                            avg_contribution = sum(self.contribution_history[i]) / len(self.contribution_history[i])
                            contribution_std = torch.tensor(self.contribution_history[i]).std().item()
                            
                            # 分析行为特征
                            behavior_features = []
                            if belief_value > 0.6:
                                behavior_features.append("高信念值")
                            if abs(contribution) > 30:
                                behavior_features.append("显著贡献")
                            if adversary_score > 0.7:
                                behavior_features.append("高对抗倾向")
                            if contribution < -20:
                                behavior_features.append("负面影响")
                            if contribution_std > 20:
                                behavior_features.append("不稳定")
                            if abs(weight_mean) > 0.8:
                                behavior_features.append("高权重")
                            
                            behavior_str = ", ".join(behavior_features) if behavior_features else "正常"
                            
                            print(f"{i:^9d} | {contribution:^8.2f}% | {belief_value:^6.4f} | {adversary_score:^10.4f} | {weight_mean:^8.4f} | {q_mean:^7.4f} | {behavior_str}")
                    
                    print("-" * 80)
                    print(f"* 贡献占比: 正值表示正面贡献，负值表示负面影响")
                    print(f"* 信念值: 越高表示越可能是对抗性智能体")
                    print(f"* 对抗性得分: 基于行为模式分析的对抗倾向")
                    print(f"* 权重均值: 智能体在混合网络中的重要性")
                    print(f"* Q值均值: 智能体的动作价值估计")
                    print("-" * 80)
                    
                    return relative_contributions, adversary_scores
                
                return None, None
                
        except Exception as e:
            print(f"\n分析智能体贡献时出错: {str(e)}")
            print(f"Debug信息:")
            print(f"weighted_qs shape: {weighted_qs.shape if weighted_qs is not None else 'None'}")
            print(f"w1 shape: {w1.shape if w1 is not None else 'None'}")
            print(f"beliefs shape: {beliefs.shape if beliefs is not None else 'None'}")
            return None, None

    def forward(self, agent_qs, states, beliefs):
        """
        前向传播过程
        
        参数:
            agent_qs: 各智能体的Q值 [batch_size, n_agents]
            states: 全局状态 [batch_size, state_dim]
            beliefs: 智能体类型的信念值 [batch_size, n_agents]
            
        返回:
            q_total: 混合后的总Q值 [batch_size, 1]
        """
        # 确保输入维度正确
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        if len(beliefs.shape) == 1:
            beliefs = beliefs.unsqueeze(0)
        if len(agent_qs.shape) == 1:
            agent_qs = agent_qs.unsqueeze(0)
            
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
        attention_concat = torch.cat([state_weights, belief_weights], dim=-1)
        attention_weights = self.attention_fusion(attention_concat)
        
        # 2. 基于信念的防御权重
        defensive_weights = 1.0 - beliefs  # 信念值越高，防御权重越低
        
        # 3. 最终权重：结合注意力权重和防御权重
        final_weights = attention_weights * defensive_weights
        
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
        
        # 分析智能体贡献
        self.analyze_agent_contributions(weighted_qs, w1, beliefs)
        
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