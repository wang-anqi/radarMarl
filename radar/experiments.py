import radar.data as data
from os.path import join
import numpy
import random
import radar.algorithm as algorithm
import radar.controller_loader as controller_loader
import os

def log(log_level, message_level, message):
    if message_level <= log_level:
        print(message)

def run_episode(episode_id, controller, params, is_adversary, training_mode=True, log_level=0, reset_episode=True):
    """运行单个episode
    
    参数:
        episode_id: episode ID
        controller: 控制器
        params: 参数字典
        is_adversary: 是否是对抗性智能体
        training_mode: 是否是训练模式
        log_level: 日志级别
        reset_episode: 是否重置环境
    """
    try:
        env = params["env"]
        path = params.get("directory", ".")
        save_summaries = params.get("save_summaries", False)
        nr_agents = params.get("nr_agents", 1)
        adversary_ids = controller.generate_adversary_ids(is_adversary)
        if reset_episode:
            observations = env.reset(adversary_ids)
        else:
            observations = env.joint_observation(adversary_ids)
        state = env.global_state()
        done = False
        time_step = 0
        state_summaries = [env.state_summary()]
        protagonist_discounted_return = 0
        protagonist_undiscounted_return = 0
        nr_protagonists = max(1.0, float(nr_agents - len(adversary_ids)))
        while not done:
            print(f"Episode {episode_id} training_mode: {training_mode}")
            joint_action = controller.policy(observations, training_mode)
            next_observations, rewards, dones, info = env.step(joint_action, adversary_ids)
            protagonist_reward = sum([float(r)/nr_protagonists for i,r in enumerate(rewards) if i not in adversary_ids])
            protagonist_discounted_return += (float(params.get("gamma", 0.99))**time_step)*protagonist_reward
            protagonist_undiscounted_return += protagonist_reward
            next_state = env.global_state()
            done = all(dones) or time_step >= int(params.get("max_episode_steps", 1000))
            state_summary = env.state_summary()
            policy_updated = False
            if training_mode:
                try:
                    policy_updated = bool(controller.update(\
                        state, observations, joint_action, rewards,\
                        next_state, next_observations, dones, is_adversary))
                except Exception as e:
                    print(f"策略更新失败: {str(e)}")
                    policy_updated = False
            state = next_state
            observations = next_observations
            state_summary["transition_info"] = info
            state_summaries.append(state_summary)
            time_step += 1
        log(log_level, 0, "{} episode {} finished:\n\tdiscounted return: {}\n\tundiscounted return: {}\n\tdomain statistics: {}"
            .format(params["domain_name"], episode_id, env.discounted_return, env.undiscounted_return, env.domain_statistic(controller.adversary_ids)))
        if save_summaries and training_mode:
            summary_filename = "episode_{}.json".format(episode_id)
            data.save_json(join(path, summary_filename), state_summaries)
            del state_summaries
        return float(protagonist_discounted_return), float(protagonist_undiscounted_return), bool(policy_updated), int(time_step)
    except Exception as e:
        print(f"Episode运行失败: {str(e)}")
        return 0.0, 0.0, False, 0

def run_test(env, nr_test_episodes, controller, params, test_adversary_ratio, log_level, is_adversary):
    training_adversary_ratio = controller.adversary_ratio # Save ratio for later training
    controller.adversary_ratio = test_adversary_ratio
    nr_protagonists = params["nr_agents"] - int(params["nr_agents"]*test_adversary_ratio)
    test_discounted_returns = []
    test_undiscounted_returns = []
    test_domain_statistics = []
    for episode_id in range(nr_test_episodes):
        run_episode("Test-{}".format(episode_id), controller, params, is_adversary, False, log_level)
        test_discounted_returns.append(env.discounted_return)
        test_undiscounted_returns.append(env.undiscounted_return)
        test_domain_statistics.append(env.domain_statistic(controller.adversary_ids))
    controller.adversary_ratio = training_adversary_ratio # Reset training ratio
    return numpy.mean(test_discounted_returns)/nr_protagonists,\
        numpy.mean(test_undiscounted_returns)/nr_protagonists,\
        numpy.mean(test_domain_statistics)/nr_protagonists

def run_default_test(env, nr_test_episodes, controller, params, log_level, is_adversary):
    return run_test(env, nr_test_episodes, controller, params, 0, log_level, is_adversary)

def run_test_suite(env, nr_test_episodes, controller, params, log_level, is_adversary):
    """运行测试套件"""
    # 适用于评估算法在“当前训练状态下”的泛化能力，不会执行策略更新
    try:
        algorithm_choice = params["algorithm_name"]
        original_adversary_ratio = controller.adversary_ratio
        result_discounted_returns = {"protagonist_mode": not is_adversary, "test_results": {}}
        result_undiscounted_returns = {"protagonist_mode": not is_adversary, "test_results": {}}
        result_domain_statistics = {"protagonist_mode": not is_adversary, "test_results": {}}
        labels = []
        
        for episode_id in range(nr_test_episodes):
            print(f"\nEpisode Test-{episode_id} training_mode: False")
            
            try:
                # 运行测试episode，使用正确的参数顺序
                d_return, u_return, policy_updated, steps = run_episode(
                    episode_id=f"Test-{episode_id}",
                    controller=controller,
                    params=params,
                    is_adversary=is_adversary,
                    training_mode=False,
                    log_level=log_level
                )
                
                # 确保返回值是数值类型
                d_return = float(d_return) if d_return is not None else 0.0
                u_return = float(u_return) if u_return is not None else 0.0
                
                # 获取domain统计信息
                try:
                    d_statistic = float(env.domain_statistic(controller.adversary_ids))
                except:
                    d_statistic = 0.0
                
                # 记录结果
                result_discounted_returns["test_results"][episode_id] = d_return
                result_undiscounted_returns["test_results"][episode_id] = u_return
                result_domain_statistics["test_results"][episode_id] = d_statistic
                
            except Exception as e:
                print(f"Episode {episode_id} 运行失败: {str(e)}")
                # 记录失败的episode的默认值
                result_discounted_returns["test_results"][episode_id] = 0.0
                result_undiscounted_returns["test_results"][episode_id] = 0.0
                result_domain_statistics["test_results"][episode_id] = 0.0
        
        return result_discounted_returns, result_undiscounted_returns, result_domain_statistics
        
    except Exception as e:
        print(f"测试套件运行失败: {str(e)}")
        # 返回默认值
        default_results = {"protagonist_mode": not is_adversary, "test_results": {0: 0.0}}
        return default_results, default_results, default_results

def run(controller, nr_episodes, params, log_level=0):
    env = params["env"]
    path = params["directory"]
    nr_test_episodes = params["nr_test_episodes"]
    test_suite = params["test_suite"]
    if test_suite is None:
        test_suite = run_default_test
    training_discounted_returns = []
    training_undiscounted_returns = []
    training_adversary_ratios = []
    test_discounted_returns = []
    test_undiscounted_returns = []
    domain_statistic = []
    test_domain_statistics = []
    is_adversary = False
    total_steps = 0  # 实际训练步数
    max_steps = params.get("nr_steps", float("inf"))  # 从参数中获取最大步数
    
    test_discounted_return, test_undiscounted_return, test_domain_statistic = \
        test_suite(env, nr_test_episodes, controller, params, log_level, is_adversary)
    test_discounted_returns.append(test_discounted_return)
    test_undiscounted_returns.append(test_undiscounted_return)
    test_domain_statistics.append(test_domain_statistic)
    nr_epoch_updates = 0
    
    for episode_id in range(nr_episodes):
        # 检查是否达到总步数限制
        if total_steps >= max_steps:
            print(f"\n达到训练步数限制 {max_steps}，停止训练")
            break
            
        protagonist_discounted_return, protagonist_undiscounted_return, policy_updated, episode_steps =\
            run_episode(episode_id, controller, params, is_adversary, True, log_level)
            
        # 更新实际训练步数
        total_steps += episode_steps
        
        if policy_updated:
            print(f"\n=====> Episode {episode_id+1} 更新完成 <=====")
            print(f"当前训练步数: {total_steps}/{max_steps}")
            print(f"本轮实际步数: {episode_steps}")
            is_adversary = not is_adversary
            test_discounted_return, test_undiscounted_return, test_domain_statistic = \
                test_suite(env, nr_test_episodes, controller, params, log_level, is_adversary)
            test_discounted_returns.append(test_discounted_return)
            test_undiscounted_returns.append(test_undiscounted_return)
            test_domain_statistics.append(test_domain_statistic)
            training_discounted_returns.append(protagonist_discounted_return)
            training_undiscounted_returns.append(protagonist_undiscounted_return)
            training_adversary_ratios.append(controller.adversary_ratio)
            nr_epoch_updates += 1
            
    print(f"\n训练结束:")
    print(f"总训练步数: {total_steps}/{max_steps}")
    print(f"完成episode数: {episode_id + 1}/{nr_episodes}")
    print(f"策略更新次数: {nr_epoch_updates}")
            
    return {
        "training_discounted_returns": training_discounted_returns,
        "training_undiscounted_returns": training_undiscounted_returns,
        "training_adversary_ratios": training_adversary_ratios,
        "test_discounted_returns": test_discounted_returns,
        "test_undiscounted_returns": test_undiscounted_returns,
        "test_domain_statistics": test_domain_statistics,
        "total_steps": total_steps,
        "completed_episodes": episode_id + 1,
        "policy_updates": nr_epoch_updates
    }
