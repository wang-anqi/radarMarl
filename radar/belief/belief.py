import torch
import torch.nn as nn
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
    """信念网络主类"""
    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        super(Belief, self).__init__()
        
        print("\n" + "="*50)
        print("Initializing Belief Network:")
        print("-"*50)
        
        # 保存参数
        self.hidden_sizes = args["hidden_sizes"]
        self.num_agents = num_agents
        self.device = device
        
        # 打印基本配置信息
        print(f"Number of agents: {self.num_agents}")
        print(f"Hidden sizes: {self.hidden_sizes}")
        print(f"Device: {self.device}")
        
        # 获取观察空间形状
        obs_shape = get_shape_from_obs_space(obs_space)
        print(f"Observation shape: {obs_shape}")
        
        # 选择并初始化特征提取器
        if len(obs_shape) == 3:
            print("Using CNN feature extractor")
            self.feature_extractor = CNNBase(args, obs_shape)
        else:
            print("Using MLP feature extractor")
            self.feature_extractor = MLPBase(args, obs_shape)
            
        # 初始化RNN层
        if args["use_recurrent_belief"]:
            print(f"Using RNN with {args['recurrent_N']} layers")
            self.rnn = RNNLayer(
                input_size=self.hidden_sizes[-1],
                hidden_size=self.hidden_sizes[-1],
                num_layers=args["recurrent_N"],
                init_method=args["initialization_method"]
            )
        else:
            print("RNN disabled")
            self.rnn = None
            
        # 初始化信念投影层
        print(f"Initializing belief projection layer with method: {args['initialization_method']}")
        self.belief_proj = BeliefProj(
            input_size=self.hidden_sizes[-1],
            num_agents=num_agents,
            init_method=args["initialization_method"],
            gain=args["gain"]
        )
        
        # 将网络移动到指定设备
        self.to(device)
        
        print("Belief Network initialization completed")
        print("="*50 + "\n")
        
    def forward(self, obs, rnn_states, masks):
        """前向传播
        Args:
            obs: 观察值
            rnn_states: RNN隐藏状态
            masks: 智能体掩码
        Returns:
            belief_probs: 每个智能体的对抗倾向概率
            new_rnn_states: 更新后的RNN隐藏状态
        """
        # 检查并转换输入
        obs = check(obs).to(self.device)
        rnn_states = check(rnn_states).to(self.device)
        masks = check(masks).to(self.device)
        
        # 提取特征
        features = self.feature_extractor(obs)
        
        # RNN处理（如果启用）
        if self.rnn is not None:
            features, new_rnn_states = self.rnn(features, rnn_states, masks)
        else:
            new_rnn_states = rnn_states
            
        # 生成信念概率
        belief_probs = self.belief_proj(features)
        
        return belief_probs, new_rnn_states