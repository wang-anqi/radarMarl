import torch
import numpy as np
import random
from enum import Enum

class AdversarialType(Enum):
    """对抗性智能体类型"""
    RANDOM = "random"           # 随机策略
    OPPOSITE = "opposite"       # 反向策略  
    SELFISH = "selfish"         # 自私策略
    NOISY = "noisy"            # 噪声策略
    Byzantine = "byzantine"     # 拜占庭错误
    LAZY = "lazy"              # 懒惰策略

class AdversarialAgent:
    """对抗性智能体实现
    
    设计多种对抗性行为模式，用于测试信念网络的识别能力
    """
    
    def __init__(self, agent_id, adversarial_type, n_actions, params=None):
        """
        初始化对抗性智能体
        
        参数:
            agent_id: 智能体ID
            adversarial_type: 对抗性类型
            n_actions: 动作空间大小
            params: 其他参数
        """
        self.agent_id = agent_id
        self.adversarial_type = adversarial_type
        self.n_actions = n_actions
        self.params = params or {}
        
        # 对抗性参数
        self.noise_level = self.params.get("noise_level", 0.3)
        self.opposite_prob = self.params.get("opposite_prob", 0.8)
        self.random_prob = self.params.get("random_prob", 0.9)
        self.selfish_weight = self.params.get("selfish_weight", 2.0)
        self.lazy_prob = self.params.get("lazy_prob", 0.6)
        
        # 历史信息追踪
        self.action_history = []
        self.reward_history = []
        self.observation_history = []
        self.step_count = 0
        
        # 行为统计
        self.behavior_stats = {
            'total_actions': 0,
            'random_actions': 0,
            'opposite_actions': 0,
            'selfish_actions': 0,
            'normal_actions': 0,
            'negative_rewards': 0,
            'action_changes': 0
        }
        
        print(f"\n初始化对抗性智能体 {agent_id}:")
        print(f"- 对抗类型: {adversarial_type.value}")
        print(f"- 动作空间: {n_actions}")
        print(f"- 噪声水平: {self.noise_level}")
        print(f"- 反向概率: {self.opposite_prob}")
        
    def get_action(self, observation, q_values=None, normal_action=None):
        """
        根据对抗性类型生成动作
        
        参数:
            observation: 当前观察
            q_values: Q值 (如果可用)
            normal_action: 正常智能体会选择的动作
            
        返回:
            action: 选择的动作
            action_info: 动作选择信息
        """
        self.step_count += 1
        action_info = {
            'type': self.adversarial_type.value,
            'step': self.step_count,
            'agent_id': self.agent_id
        }
        
        # 保存观察历史
        self.observation_history.append(observation)
        if len(self.observation_history) > 100:  # 限制历史长度
            self.observation_history.pop(0)
        
        # 根据对抗类型选择动作
        if self.adversarial_type == AdversarialType.RANDOM:
            action = self._random_action()
            action_info['strategy'] = 'random'
            self.behavior_stats['random_actions'] += 1
            
        elif self.adversarial_type == AdversarialType.OPPOSITE:
            action = self._opposite_action(q_values, normal_action)
            action_info['strategy'] = 'opposite'
            self.behavior_stats['opposite_actions'] += 1
            
        elif self.adversarial_type == AdversarialType.SELFISH:
            action = self._selfish_action(observation, q_values)
            action_info['strategy'] = 'selfish'
            self.behavior_stats['selfish_actions'] += 1
            
        elif self.adversarial_type == AdversarialType.NOISY:
            action = self._noisy_action(q_values, normal_action)
            action_info['strategy'] = 'noisy'
            
        elif self.adversarial_type == AdversarialType.BYZANTINE:
            action = self._byzantine_action(observation)
            action_info['strategy'] = 'byzantine'
            
        elif self.adversarial_type == AdversarialType.LAZY:
            action = self._lazy_action()
            action_info['strategy'] = 'lazy'
            
        else:
            # 默认随机动作
            action = self._random_action()
            action_info['strategy'] = 'default_random'
            self.behavior_stats['random_actions'] += 1
        
        # 更新行为统计
        self.behavior_stats['total_actions'] += 1
        
        # 记录动作变化
        if len(self.action_history) > 0 and action != self.action_history[-1]:
            self.behavior_stats['action_changes'] += 1
        
        # 保存动作历史
        self.action_history.append(action)
        if len(self.action_history) > 100:
            self.action_history.pop(0)
        
        return action, action_info
    
    def _random_action(self):
        """随机策略：完全随机选择动作"""
        if random.random() < self.random_prob:
            return random.randint(0, self.n_actions - 1)
        else:
            # 偶尔选择固定动作以增加识别难度
            return 0
    
    def _opposite_action(self, q_values, normal_action):
        """反向策略：选择与最优动作相反的动作"""
        if normal_action is not None and random.random() < self.opposite_prob:
            # 选择Q值最低的动作
            if q_values is not None:
                if isinstance(q_values, torch.Tensor):
                    worst_action = q_values.argmin().item()
                else:
                    worst_action = np.argmin(q_values)
                return worst_action
            else:
                # 如果没有Q值，选择与正常动作不同的动作
                opposite_actions = [a for a in range(self.n_actions) if a != normal_action]
                return random.choice(opposite_actions) if opposite_actions else normal_action
        else:
            return random.randint(0, self.n_actions - 1)
    
    def _selfish_action(self, observation, q_values):
        """自私策略：只考虑自己的利益，不考虑团队合作"""
        # 简化实现：倾向于选择可能对自己有利但对团队不利的动作
        if q_values is not None:
            # 在最优动作周围添加自私偏置
            if isinstance(q_values, torch.Tensor):
                best_action = q_values.argmax().item()
            else:
                best_action = np.argmax(q_values)
            
            # 有概率选择次优动作，模拟自私行为
            if random.random() < 0.7:
                suboptimal_actions = [a for a in range(self.n_actions) if a != best_action]
                return random.choice(suboptimal_actions) if suboptimal_actions else best_action
            else:
                return best_action
        else:
            return random.randint(0, self.n_actions - 1)
    
    def _noisy_action(self, q_values, normal_action):
        """噪声策略：在正常动作基础上添加噪声"""
        if normal_action is not None and random.random() > self.noise_level:
            return normal_action
        else:
            # 添加噪声：随机选择其他动作
            if normal_action is not None:
                noise_actions = [a for a in range(self.n_actions) if a != normal_action]
                return random.choice(noise_actions) if noise_actions else normal_action
            else:
                return random.randint(0, self.n_actions - 1)
    
    def _byzantine_action(self, observation):
        """拜占庭错误：模拟系统故障或恶意行为"""
        # 模拟间歇性故障
        if self.step_count % 10 < 3:  # 30%的时间表现异常
            return random.randint(0, self.n_actions - 1)
        elif self.step_count % 10 < 6:  # 30%的时间保持静止
            return 0  # 假设0是静止动作
        else:  # 40%的时间正常行为
            return random.randint(0, self.n_actions - 1)
    
    def _lazy_action(self):
        """懒惰策略：倾向于选择最少工作量的动作"""
        if random.random() < self.lazy_prob:
            return 0  # 假设0是最少工作量的动作
        else:
            return random.randint(0, self.n_actions - 1)
    
    def process_reward(self, reward):
        """处理奖励信息，用于分析对抗效果"""
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        
        if reward < 0:
            self.behavior_stats['negative_rewards'] += 1
    
    def get_adversarial_features(self):
        """获取对抗性特征，用于信念网络训练"""
        if self.behavior_stats['total_actions'] == 0:
            return torch.zeros(8)  # 返回默认特征
        
        features = [
            # 1. 随机动作比例
            self.behavior_stats['random_actions'] / max(1, self.behavior_stats['total_actions']),
            
            # 2. 反向动作比例  
            self.behavior_stats['opposite_actions'] / max(1, self.behavior_stats['total_actions']),
            
            # 3. 自私动作比例
            self.behavior_stats['selfish_actions'] / max(1, self.behavior_stats['total_actions']),
            
            # 4. 动作变化频率
            self.behavior_stats['action_changes'] / max(1, self.behavior_stats['total_actions'] - 1),
            
            # 5. 负奖励比例
            self.behavior_stats['negative_rewards'] / max(1, len(self.reward_history)),
            
            # 6. 平均奖励（如果有历史）
            np.mean(self.reward_history) if self.reward_history else 0.0,
            
            # 7. 奖励方差
            np.var(self.reward_history) if len(self.reward_history) > 1 else 0.0,
            
            # 8. 对抗类型编码
            self._get_type_encoding()
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_type_encoding(self):
        """获取对抗类型的数值编码"""
        type_map = {
            AdversarialType.RANDOM: 0.1,
            AdversarialType.OPPOSITE: 0.2,
            AdversarialType.SELFISH: 0.3,
            AdversarialType.NOISY: 0.4,
            AdversarialType.BYZANTINE: 0.5,
            AdversarialType.LAZY: 0.6
        }
        return type_map.get(self.adversarial_type, 0.0)
    
    def print_behavior_stats(self):
        """打印行为统计信息"""
        total = max(1, self.behavior_stats['total_actions'])
        print(f"\n智能体 {self.agent_id} 行为统计:")
        print(f"- 总动作数: {self.behavior_stats['total_actions']}")
        print(f"- 随机动作: {self.behavior_stats['random_actions']} ({self.behavior_stats['random_actions']/total*100:.1f}%)")
        print(f"- 反向动作: {self.behavior_stats['opposite_actions']} ({self.behavior_stats['opposite_actions']/total*100:.1f}%)")
        print(f"- 自私动作: {self.behavior_stats['selfish_actions']} ({self.behavior_stats['selfish_actions']/total*100:.1f}%)")
        print(f"- 动作变化: {self.behavior_stats['action_changes']} ({self.behavior_stats['action_changes']/max(1,total-1)*100:.1f}%)")
        print(f"- 负奖励次数: {self.behavior_stats['negative_rewards']}")
        if self.reward_history:
            print(f"- 平均奖励: {np.mean(self.reward_history):.3f}")
            print(f"- 奖励标准差: {np.std(self.reward_history):.3f}")

class AdversarialAgentManager:
    """对抗性智能体管理器"""
    
    def __init__(self, n_agents, n_actions, adversarial_config):
        """
        初始化对抗性智能体管理器
        
        参数:
            n_agents: 总智能体数量
            n_actions: 动作空间大小
            adversarial_config: 对抗性配置
                {
                    'adversary_ids': [0, 2],  # 对抗性智能体ID列表
                    'adversary_types': ['random', 'opposite'],  # 对应的对抗类型
                    'params': {...}  # 其他参数
                }
        """
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.adversarial_config = adversarial_config
        
        # 解析配置
        self.adversary_ids = adversarial_config.get('adversary_ids', [])
        adversary_types = adversarial_config.get('adversary_types', [])
        self.params = adversarial_config.get('params', {})
        
        # 创建对抗性智能体
        self.adversarial_agents = {}
        
        for i, agent_id in enumerate(self.adversary_ids):
            if i < len(adversary_types):
                adv_type_str = adversary_types[i]
                adv_type = AdversarialType(adv_type_str)
            else:
                adv_type = AdversarialType.RANDOM  # 默认类型
            
            self.adversarial_agents[agent_id] = AdversarialAgent(
                agent_id=agent_id,
                adversarial_type=adv_type,
                n_actions=n_actions,
                params=self.params
            )
        
        print(f"\n对抗性智能体管理器初始化完成:")
        print(f"- 总智能体数: {n_agents}")
        print(f"- 对抗性智能体数: {len(self.adversary_ids)}")
        print(f"- 对抗性智能体ID: {self.adversary_ids}")
    
    def is_adversarial(self, agent_id):
        """检查智能体是否为对抗性"""
        return agent_id in self.adversarial_agents
    
    def get_action(self, agent_id, observation, q_values=None, normal_action=None):
        """获取指定智能体的动作"""
        if agent_id in self.adversarial_agents:
            return self.adversarial_agents[agent_id].get_action(
                observation, q_values, normal_action
            )
        else:
            return normal_action, {'type': 'normal', 'agent_id': agent_id}
    
    def process_rewards(self, rewards):
        """处理所有智能体的奖励"""
        for agent_id, reward in enumerate(rewards):
            if agent_id in self.adversarial_agents:
                self.adversarial_agents[agent_id].process_reward(reward)
    
    def get_all_adversarial_features(self):
        """获取所有对抗性智能体的特征"""
        features = {}
        for agent_id, agent in self.adversarial_agents.items():
            features[agent_id] = agent.get_adversarial_features()
        return features
    
    def print_all_stats(self):
        """打印所有对抗性智能体的统计信息"""
        print("\n" + "="*60)
        print("对抗性智能体行为统计报告")
        print("="*60)
        for agent_id, agent in self.adversarial_agents.items():
            agent.print_behavior_stats()
        print("="*60)

def create_adversarial_config(n_agents, adversary_ratio=0.3, strategy='mixed'):
    """
    创建对抗性配置的辅助函数
    
    参数:
        n_agents: 总智能体数量
        adversary_ratio: 对抗者比例
        strategy: 配置策略
            - 'first': 前几个智能体为对抗性
            - 'random': 随机选择对抗性智能体
            - 'mixed': 混合不同类型的对抗性智能体
            - 'single': 只有一个对抗性智能体
    
    返回:
        adversarial_config: 配置字典
    """
    n_adversaries = max(1, int(n_agents * adversary_ratio))
    
    if strategy == 'first':
        adversary_ids = list(range(n_adversaries))
        adversary_types = ['random'] * n_adversaries
        
    elif strategy == 'random':
        adversary_ids = random.sample(range(n_agents), n_adversaries)
        adversary_types = random.choices(
            ['random', 'opposite', 'selfish', 'noisy'], 
            k=n_adversaries
        )
        
    elif strategy == 'mixed':
        adversary_ids = list(range(n_adversaries))
        type_options = ['random', 'opposite', 'selfish', 'noisy', 'byzantine', 'lazy']
        adversary_types = [type_options[i % len(type_options)] for i in range(n_adversaries)]
        
    elif strategy == 'single':
        adversary_ids = [0]  # 第一个智能体为对抗性
        adversary_types = ['opposite']  # 使用反向策略
        
    else:
        raise ValueError(f"未知的策略: {strategy}")
    
    config = {
        'adversary_ids': adversary_ids,
        'adversary_types': adversary_types,
        'params': {
            'noise_level': 0.3,
            'opposite_prob': 0.8,
            'random_prob': 0.9,
            'selfish_weight': 2.0,
            'lazy_prob': 0.6
        }
    }
    
    print(f"\n创建对抗性配置:")
    print(f"- 策略: {strategy}")
    print(f"- 对抗者ID: {adversary_ids}")
    print(f"- 对抗者类型: {adversary_types}")
    
    return config