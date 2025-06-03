import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_shape_from_obs_space(obs_space):
    """获取观察空间的形状"""
    if isinstance(obs_space, tuple):
        return list(obs_space)
    elif hasattr(obs_space, 'shape'):
        return list(obs_space.shape)
    else:
        raise NotImplementedError

def check(x):
    """检查并转换输入为tensor"""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float)

class MLPBase(nn.Module):
    """MLP基础网络"""
    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()
        
        input_size = obs_shape[0]
        hidden_sizes = args["hidden_sizes"]
        activation = args["activation_func"]
        
        # 构建MLP层
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            if activation == "ReLU":
                layers.append(nn.ReLU())
            elif activation == "Tanh":
                layers.append(nn.Tanh())
            last_size = size
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)

class CNNBase(nn.Module):
    """CNN基础网络"""
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()
        
        # 根据输入尺寸调整CNN配置
        if min(obs_shape[1:]) < 8:  # 如果输入尺寸小于8x8
            # 使用更小的卷积核和步长
            self.cnn = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=2, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
        else:
            # 原始CNN配置（用于大尺寸输入）
            self.cnn = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
        
        # 计算CNN输出大小
        with torch.no_grad():
            sample = torch.zeros(1, *obs_shape)
            cnn_out = self.cnn(sample)
            cnn_out_size = cnn_out.shape[1]
        
        # 添加全连接层
        self.fc = nn.Linear(cnn_out_size, args["hidden_sizes"][-1])
        
        print(f"\nCNN Network Structure:")
        print(f"Input shape: {obs_shape}")
        print(f"CNN output size: {cnn_out_size}")
        print(f"Final output size: {args['hidden_sizes'][-1]}\n")
        
    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

class RNNLayer(nn.Module):
    """RNN层"""
    def __init__(self, input_size, hidden_size, num_layers, init_method="orthogonal"):
        super(RNNLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 初始化参数
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                if init_method == "orthogonal":
                    nn.init.orthogonal_(param)
                elif init_method == "xavier":
                    nn.init.xavier_uniform_(param)
                    
    def forward(self, x, hxs, masks):
        """
        Args:
            x: [batch_size, feature_size]
            hxs: [batch_size, hidden_size]
            masks: [batch_size, 1]
        Returns:
            x: [batch_size, hidden_size]
            hxs: [batch_size, hidden_size]
        """
        # 调整输入维度
        x = x.unsqueeze(1)  # [batch_size, 1, feature_size]
        
        # 调整隐藏状态维度
        B = x.size(0)  # batch_size
        hxs = hxs.view(B, self.hidden_size)  # 确保形状正确
        hxs = hxs.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_size]
        
        # 应用masks
        hxs = hxs * masks.view(1, -1, 1)
        
        # 前向传播
        x, hxs = self.gru(x, hxs)
        
        # 调整输出维度
        x = x.squeeze(1)  # [batch_size, hidden_size]
        hxs = hxs.mean(0)  # [batch_size, hidden_size]，取平均值作为新的隐藏状态
        
        return x, hxs

class BeliefProj(nn.Module):
    """信念投影层"""
    def __init__(self, input_size, num_agents, init_method="orthogonal", gain=0.01):
        super(BeliefProj, self).__init__()
        
        self.fc = nn.Linear(input_size, num_agents)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化参数
        if init_method == "orthogonal":
            nn.init.orthogonal_(self.fc.weight, gain=gain)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)

class Belief(nn.Module):
    """信念网络，用于评估智能体的可信度"""
    
    def __init__(self, args, obs_space, action_space, num_agents, device):
        super(Belief, self).__init__()
        
        self.args = args
        # 计算观察空间维度
        if isinstance(obs_space, (tuple, list)):
            self.obs_dim = np.prod(obs_space)
        elif hasattr(obs_space, 'shape'):
            self.obs_dim = np.prod(obs_space.shape)
        else:
            self.obs_dim = obs_space
            
        self.action_dim = action_space.n
        self.num_agents = num_agents
        self.device = device
        
        print("\n初始化信念网络:")
        print(f"观察空间维度: {self.obs_dim}")
        print(f"动作空间维度: {self.action_dim}")
        print(f"智能体数量: {self.num_agents}")
        
        # 网络参数
        self.hidden_dim = 128
        
        # 使用全连接层处理输入
        self.input_dim = self.obs_dim  # 使用实际的观察空间维度
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # RNN层
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        
        # 输出层
        self.fc_out = nn.Linear(self.hidden_dim, 1)
        
        # 将所有模块移动到指定设备
        self.to(device)
        
        # 初始化参数
        self.init_parameters()
        
        print(f"网络结构:")
        print(f"输入维度: {self.input_dim}")
        print(f"输入层: {self.input_dim} -> {self.hidden_dim}")
        print(f"隐藏层: {self.hidden_dim} -> {self.hidden_dim}")
        print(f"RNN层: {self.hidden_dim} <-> {self.hidden_dim}")
        print(f"输出层: {self.hidden_dim} -> 1")
        print("-" * 50)
        
    def init_parameters(self):
        """初始化网络参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def get_padded_belief_values(self, belief_values, target_num_agents):
        """获取填充后的信念值，确保输出维度匹配目标智能体数量
        
        参数:
            belief_values: 原始信念值 [batch_size, current_num_agents]
            target_num_agents: 目标智能体数量
            
        返回:
            padded_beliefs: 填充后的信念值 [batch_size, target_num_agents]
        """
        batch_size, current_num_agents = belief_values.size()
        if current_num_agents == target_num_agents:
            return belief_values
            
        # 创建填充后的信念值张量
        padded_beliefs = torch.zeros(batch_size, target_num_agents, device=self.device)
        
        # 复制有效的信念值
        num_agents_to_copy = min(current_num_agents, target_num_agents)
        padded_beliefs[:, :num_agents_to_copy] = belief_values[:, :num_agents_to_copy]
        
        # 如果目标智能体数量更多，将剩余位置设置为默认值（0.5表示中性信念）
        if target_num_agents > current_num_agents:
            padded_beliefs[:, current_num_agents:] = 0.5
            
        return padded_beliefs

    def forward(self, inputs, hidden_states, masks=None):
        """前向传播
        
        参数:
            inputs: 输入数据，支持多种维度:
                   2D: [batch_size * num_agents, input_dim]
                   3D: [batch_size, num_agents, input_dim]
                   4D: [batch_size, num_agents, height, width]
            hidden_states: RNN隐藏状态
            masks: 智能体有效性掩码
            
        返回:
            belief_values: 信念值 [batch_size, num_agents]
            new_hidden_states: 新的RNN隐藏状态
        """
        try:
            # 确保inputs是tensor并且在正确的设备上
            if isinstance(inputs, list):
                inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            elif isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float().to(self.device)
            elif isinstance(inputs, torch.Tensor):
                inputs = inputs.float().to(self.device)
            
            # 获取batch_size和num_agents
            original_num_agents = self.num_agents  # 保存原始智能体数量
            if inputs.dim() == 4:  # [batch_size, num_agents, height, width]
                batch_size, num_agents, height, width = inputs.size()
                # 展平空间维度
                inputs = inputs.view(batch_size, num_agents, height * width)
                print(f"处理4D输入: 原始形状 {[batch_size, num_agents, height, width]} -> 展平后 {inputs.shape}")
            elif inputs.dim() == 3:  # [batch_size, num_agents, input_dim]
                batch_size, num_agents, _ = inputs.size()
            elif inputs.dim() == 2:  # [batch_size * num_agents, input_dim]
                total_agents = inputs.size(0)
                num_agents = self.num_agents
                batch_size = total_agents // num_agents
                inputs = inputs.view(batch_size, num_agents, -1)
            else:
                raise ValueError(f"不支持的输入维度: {inputs.dim()}, 输入形状: {inputs.shape}")
            
            print(f"\n数据维度信息:")
            print(f"Batch size: {batch_size}")
            print(f"Number of agents: {num_agents}")
            print(f"Original number of agents: {original_num_agents}")
            print(f"Input shape after reshape: {inputs.shape}")
            
            # 处理掩码
            if masks is not None:
                masks = masks.to(self.device)
                # 创建新的掩码，确保维度正确
                new_masks = torch.ones(batch_size, num_agents, 1, device=self.device)
                if masks.size(1) > num_agents:
                    # 如果掩码的智能体数量过多，只取需要的部分
                    masks = masks[:, :num_agents, :]
                print(f"调整掩码维度: 从 {masks.shape} 到 [batch_size, num_agents, 1]")
                # 扩展掩码到正确的batch_size
                if masks.size(0) == 1:
                    masks = masks.expand(batch_size, -1, -1)
                # 将有效的掩码部分复制到新掩码中
                new_masks[:, :masks.size(1), :] = masks
                masks = new_masks
            
            # 重塑输入以适应处理
            flattened_input = inputs.view(-1, inputs.size(-1))  # [batch_size * num_agents, input_dim]
            
            # 检查输入维度并更新网络如果需要
            if flattened_input.size(-1) != self.input_dim:
                print(f"检测到输入维度变化: {flattened_input.size(-1)} (之前: {self.input_dim})")
                self.input_dim = flattened_input.size(-1)
                self.fc1 = nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
                self.init_parameters()
                print(f"已更新网络结构，新的输入维度: {self.input_dim}")
            
            # 处理输入
            x = F.relu(self.fc1(flattened_input))  # [batch_size * num_agents, hidden_dim]
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            
            # 重塑数据以适应RNN
            x = x.view(batch_size * num_agents, 1, self.hidden_dim)  # 添加时间维度
            
            # RNN处理
            if hidden_states is None:
                hidden_states = torch.zeros(1, batch_size * num_agents, self.hidden_dim, device=self.device)
            else:
                # 确保hidden_states维度正确
                if hidden_states.size(1) != batch_size * num_agents:
                    print(f"调整隐藏状态维度: 从 {hidden_states.shape} 到 [1, {batch_size * num_agents}, {self.hidden_dim}]")
                    hidden_states = torch.zeros(1, batch_size * num_agents, self.hidden_dim, device=self.device)
            
            hidden_states = hidden_states.contiguous()
            x, new_hidden_states = self.rnn(x, hidden_states)
            
            # 输出层
            x = x[:, -1]  # 取最后一个时间步
            belief_values = torch.sigmoid(self.fc_out(x))
            belief_values = belief_values.view(batch_size, num_agents)
            
            # 应用掩码（如果提供）
            if masks is not None:
                print(f"应用掩码: belief_values shape: {belief_values.shape}, masks shape: {masks.shape}")
                belief_values = belief_values * masks.squeeze(-1)
            
            # 确保输出维度匹配原始智能体数量
            if num_agents != original_num_agents:
                print(f"调整信念值维度: 从 {belief_values.shape} 到 [batch_size, {original_num_agents}]")
                belief_values = self.get_padded_belief_values(belief_values, original_num_agents)
            
            return belief_values, new_hidden_states
            
        except Exception as e:
            print(f"\n信念网络前向传播出错: {str(e)}")
            print(f"输入类型: {type(inputs)}")
            print(f"输入形状: {inputs.shape if hasattr(inputs, 'shape') else '未知'}")
            print(f"输入维度: {self.input_dim}")
            print(f"隐藏状态类型: {type(hidden_states)}")
            print(f"隐藏状态形状: {hidden_states.shape if hidden_states is not None and hasattr(hidden_states, 'shape') else 'None'}")
            print(f"掩码类型: {type(masks)}")
            print(f"掩码形状: {masks.shape if masks is not None and hasattr(masks, 'shape') else 'None'}")
            return None, None