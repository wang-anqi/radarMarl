import random
import numpy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from radar.utils import get_param_or_default
from radar.agents.ppo import PPOLearner

class PPOMIXLearner(PPOLearner):

    def __init__(self, params):
        self.global_input_shape = params["global_observation_shape"]
        super(PPOMIXLearner, self).__init__(params)
        self.central_q_learner = params["central_q_learner"]
        self.last_q_loss = 0

    def value_update(self, minibatch_data, is_adversary):
        batch_size = minibatch_data["states"].size(0)
        print("Batch size (minibatch_data['states'].size(0)):", batch_size)
        self.central_q_learner.zero_actions = torch.zeros(batch_size, dtype=torch.long).unsqueeze(1)
        nr_agents = self.get_nr_protagonists()

         # 根据是否为对手智能体选择相应的 returns 数据
        if not is_adversary:
            returns = minibatch_data["pro_returns"]
        else:
            returns = minibatch_data["adv_returns"]

        # 确保 returns 的形状与 nr_agents 兼容
        total_elements = returns.numel()
        
        if total_elements % nr_agents != 0:
            # 计算需要填充的元素数以使其成为 nr_agents 的倍数
            remainder = total_elements % nr_agents
            padding_elements = nr_agents - remainder
            # 填充 zeros 使元素数能够被 nr_agents 整除
            padding = torch.zeros(padding_elements, dtype=returns.dtype, device=returns.device)
            # 拼接填充后的 returns
            returns = torch.cat((returns, padding))

        # 现在 reshape 为 (-1, nr_agents)
        returns = returns.view(-1, nr_agents)
        print("--------------可以被nr_agents整除 - returns.shape:", returns.shape)
        # if not is_adversary:
        #     returns = minibatch_data["pro_returns"].view(-1, nr_agents)
        # else:
        #     returns = minibatch_data["adv_returns"].view(-1, nr_agents)
        # print("-------------------------Shape of returns after view operation:", returns.shape)
        #  确保 returns 的行数与 batch_size 一致
        current_rows, current_cols = returns.shape
        if current_rows < batch_size:
        # 如果 returns 的行数少于 batch_size，填充缺失行
            padding_rows = batch_size - current_rows
            padding = torch.zeros((padding_rows, current_cols), dtype=returns.dtype, device=returns.device)
            returns = torch.cat((returns, padding), dim=0)
        elif current_rows > batch_size:
        # 如果 returns 的行数多于 batch_size，进行截断
            returns = returns[:batch_size]
        print("-----------和batch_size大小一致 - returns.shape:", returns.shape)    
        print("------------zero_actions shape:", self.central_q_learner.zero_actions.shape)

        # 在第 1 维上使用 zero_actions 进行 gather 操作，提取对应的返回值
        returns = returns.gather(1, self.central_q_learner.zero_actions).squeeze()
        # 打印 zero_actions 的大小

        returns /= self.nr_agents
        returns *= nr_agents
        assert returns.size(0) == batch_size
        for _ in range(self.nr_epochs):
            self.last_q_loss = self.central_q_learner.train_step_with(minibatch_data, is_adversary, returns, nr_agents)

    def policy_update(self, minibatch_data, optimizer, is_adversary):
        if is_adversary:
            old_probs = minibatch_data["adv_action_probs"]
            histories = minibatch_data["adv_histories"]
            actions = minibatch_data["adv_actions"]
            returns = minibatch_data["adv_returns"]
        else:
            old_probs = minibatch_data["pro_action_probs"]
            histories = minibatch_data["pro_histories"]
            actions = minibatch_data["pro_actions"]
            returns = minibatch_data["pro_returns"]
        action_probs, expected_values = self.policy_net(histories, is_adversary)
        expected_Q_values = self.central_q_learner.policy_net(histories, is_adversary).detach()
        policy_losses = []
        value_losses = []
        for probs, action, value, Q_values, old_prob, R in\
            zip(action_probs, actions, expected_values, expected_Q_values, old_probs, returns):
            baseline = sum(probs*Q_values)
            baseline = baseline.detach()
            advantage = R - baseline
            policy_losses.append(self.policy_loss(advantage, probs, action, old_prob))
            value_losses.append(F.mse_loss(value[0], Q_values[action]))
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return True
