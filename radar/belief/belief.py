import copy
import torch
import torch.nn as nn
from eir_mappo.util.util import check
from eir_mappo.model.cnn import CNNBase
from eir_mappo.model.mlp import MLPBase, BeliefProj
from eir_mappo.model.rnn import RNNLayer
from eir_mappo.util.util import get_shape_from_obs_space


class Belief(nn.Module):
    """
    belief network类用于HAPPO。根据观察输出动作。
    :param args: (argparse.Namespace) 包含相关模型信息的参数
    :param obs_space: (gym.Space) 观察空间
    :param action_space: (gym.Space) 动作空间 
    :param device: (torch.device) 指定运行设备(cpu/gpu)
    """

    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        super(Belief, self).__init__()
        # 从args中获取各种参数配置
        self.hidden_sizes = args["hidden_sizes"]  # 隐藏层大小
        self.args = args  # 存储所有参数
        self.gain = args["gain"]  # 增益参数
        self.initialization_method = args["initialization_method"]  # 初始化方法
        self.use_policy_active_masks = args["use_policy_active_masks"]  # 是否使用策略激活掩码
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]  # 是否使用简单循环策略
        self.use_recurrent_policy = args["use_recurrent_policy"]  # 是否使用循环策略
        self.use_recurrent_belief = args["use_recurrent_belief"]  # 是否使用循环信念
        self.activation_func = args["activation_func"]  # 激活函数
        self.recurrent_N = args["recurrent_N"]  # 循环层数
        self.belief_option = args["belief_option"]  # 信念选项
        self.hard_belief_thres = args["hard_belief_thres"]  # 硬信念阈值
        self.tpdv = dict(dtype=torch.float32, device=device)  # 张量参数字典
        self.num_agents = num_agents  # 智能体数量
        self.obs_space = obs_space  # 观察空间
    
        # 复制并修改观察空间形状
        obs_shape = copy.deepcopy(get_shape_from_obs_space(self.obs_space))
        obs_shape[0] -= self.num_agents  # 减去智能体数量
        # 根据观察空间维度选择基础网络类型
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)  # 初始化基础网络

        # 如果使用任意一种循环策略,初始化RNN层
        if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_belief:
            self.rnn = RNNLayer(self.hidden_sizes[-1], self.hidden_sizes[-1],
                                self.recurrent_N, self.initialization_method)
        
        # 初始化信念投影层
        self.belief = BeliefProj(self.hidden_sizes[-1], self.num_agents, self.initialization_method, self.gain)
        
        # 将模型移至指定设备
        self.to(device)

    def forward(self, obs, belief_rnn_states, masks):
        """
        根据输入计算信念。
        :param obs: (np.ndarray / torch.Tensor) 输入网络的观察
        :param rnn_states: (np.ndarray / torch.Tensor) RNN网络的隐藏状态
        :param masks: (np.ndarray / torch.Tensor) 掩码张量,表示是否应将隐藏状态重新初始化为零
        :param available_actions: (np.ndarray / torch.Tensor) 表示智能体可用的动作
        :param deterministic: (bool) 是否从动作分布中采样或返回众数

        :return belief: (torch.Tensor) 要采取的动作
        :return action_log_probs: (torch.Tensor) 所采取动作的对数概率
        :return rnn_states: (torch.Tensor) 更新后的RNN隐藏状态
        """
        # 将输入转换为张量并移至指定设备
        obs = check(obs).to(**self.tpdv)
        belief_rnn_states = check(belief_rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 通过基础网络提取特征
        belief_features = self.base(obs)

        # 如果使用循环网络,更新隐藏状态
        if self.use_naive_recurrent_policy or self.use_recurrent_policy or self.use_recurrent_belief:
            belief_features, belief_rnn_states = self.rnn(
                belief_features, belief_rnn_states, masks)

        # 计算信念
        belief = self.belief(belief_features)

        # 如果使用硬信念,根据阈值二值化
        if self.belief_option == 'hard':
            belief = torch.where(belief > self.hard_belief_thres, 1.0, 0.0)

        return belief, belief_rnn_states

