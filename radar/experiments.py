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
    env = params["env"]
    path = params["directory"]
    save_summaries = params["save_summaries"]
    nr_agents = params["nr_agents"]
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
    nr_protagonists = 1.0*(nr_agents-len(adversary_ids))
    while not done:
        joint_action = controller.policy(observations, training_mode)
        next_observations, rewards, dones, info = env.step(joint_action, adversary_ids)
        protagonist_reward = sum([r/nr_protagonists for i,r in enumerate(rewards) if i not in adversary_ids])
        protagonist_discounted_return += (params["gamma"]**time_step)*protagonist_reward
        protagonist_undiscounted_return += protagonist_reward
        next_state = env.global_state()
        done = not [d for i,d in enumerate(dones) if (not d) and (i not in adversary_ids)]
        state_summary = env.state_summary()
        policy_updated = False
        if training_mode:
            policy_updated = controller.update(\
                state, observations, joint_action, rewards,\
                next_state, next_observations, dones, is_adversary)
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
    return protagonist_discounted_return, protagonist_undiscounted_return, policy_updated, time_step

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
    algorithm_choice = params["algorithm_name"]
    test_adversary_ratios = params["test_adversary_ratios"]
    original_adversary_ratio = controller.adversary_ratio
    result_discounted_returns = {"protagonist_mode": not is_adversary, "test_results": {}}
    result_undiscounted_returns = {"protagonist_mode": not is_adversary, "test_results": {}}
    result_domain_statistics = {"protagonist_mode": not is_adversary, "test_results": {}}
    labels = []
    
    # 对于 BELIEF_QMIX，只测试自身
    if algorithm_choice == "BELIEF_QMIX":
        test_agent = controller  # 直接使用当前控制器
        label = "BELIEF_QMIX"
        labels.append(label)
        d_return, u_return, d_statistic = \
            run_test(env, nr_test_episodes, test_agent, params, test_agent.adversary_ratio, -1, is_adversary)
        
        result_discounted_returns["test_results"][label] = d_return
        result_undiscounted_returns["test_results"][label] = u_return
        result_domain_statistics["test_results"][label] = d_statistic
        
        print("-- R =", test_agent.adversary_ratio,"|",label,"|",test_agent.adversary_ratio)
        
        if not is_adversary:
            test_prefix = "Protagonist"
        else:
            test_prefix = "Adversary"
            
        log(log_level, 0, "{} episode {} finished:\n\tdiscounted return: {}\n\tundiscounted return: {}\n\tdomain statistics: {}"
            .format(params["domain_name"], "{} Test {} vs. {} ({})"\
                .format(test_prefix, algorithm_choice, label, test_agent.adversary_ratio),\
                result_discounted_returns["test_results"][label],\
                result_undiscounted_returns["test_results"][label],\
                result_domain_statistics["test_results"][label]))
                
        return result_discounted_returns, result_undiscounted_returns, result_domain_statistics
    
    # 其他算法的原有逻辑
    for test_algorithm in params["test_algorithms"]:
        if is_adversary:
            test_adversary_ratios = [ratio for ratio in test_adversary_ratios if ratio > 0]
        for adversary_ratio in test_adversary_ratios:
            if adversary_ratio >= 0 or test_algorithm == algorithm_choice or algorithm_choice in ["IAC", "AC-QMIX", "PPO", "COMA", "PPO-QMIX", "MADDPG", "M3DDPG"]:
                discounted_return = 0
                undiscounted_return = 0
                domain_statistic = 0
                params["adversary_ratio"] = max(adversary_ratio, 0)
                paths = controller_loader.get_paths(params["test_directory"], test_algorithm, params)
                if adversary_ratio >= 0:
                     nr_test_episodes = len(paths)
                for i in range(nr_test_episodes):
                    factor = 1
                    if adversary_ratio < 0:
                        label = "cooperative"
                        test_agent = algorithm.make(algorithm_choice, params)
                        test_agent.adversary_ratio = 0
                        test_agent.policy_net.protagonist_net = controller.policy_net.protagonist_net
                        test_agent.policy_net.adversary_net = controller.policy_net.adversary_net
                    else:
                        label = "algorithm-{}_ratio-{}".format(test_algorithm, adversary_ratio)
                        other_agents = controller_loader.load_agents(paths[i], test_algorithm, params)
                        if not is_adversary:
                            test_agent = controller_loader.combine_agents(\
                                    controller, other_agents, test_algorithm, params)
                        else:
                            test_agent = controller_loader.combine_agents(\
                                    other_agents, controller, test_algorithm, params)
                            factor = -1
                    if i == 0 and adversary_ratio >= 0:
                        labels.append(label)
                    d_return, u_return, d_statistic = \
                        run_test(env, 1, test_agent, params, test_agent.adversary_ratio, -1, is_adversary)
                    discounted_return += d_return
                    undiscounted_return += u_return
                    domain_statistic += d_statistic
                print("-- R =", test_agent.adversary_ratio,"|",label,"|",test_agent.adversary_ratio)
                result_discounted_returns["test_results"][label] = factor*discounted_return*1.0/nr_test_episodes
                result_undiscounted_returns["test_results"][label] = factor*undiscounted_return*1.0/nr_test_episodes
                result_domain_statistics["test_results"][label] = factor*domain_statistic*1.0/nr_test_episodes
                if not is_adversary:
                    test_prefix = "Protagonist"
                else:
                    test_prefix = "Adversary"
                log(log_level, 0, "{} episode {} finished:\n\tdiscounted return: {}\n\tundiscounted return: {}\n\tdomain statistics: {}"
                .format(params["domain_name"], "{} Test {} vs. {} ({})"\
                    .format(test_prefix,algorithm_choice,test_algorithm, adversary_ratio),\
                    result_discounted_returns["test_results"][label],\
                    result_undiscounted_returns["test_results"][label],\
                        result_domain_statistics["test_results"][label]))
    params["adversary_ratio"] = original_adversary_ratio
    return result_discounted_returns, result_undiscounted_returns, result_domain_statistics

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
