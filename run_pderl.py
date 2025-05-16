import numpy as np, os, time, random
from pderl import utils, agents
import gymnasium as gym, torch
import argparse
import pickle
import json
from pderl.parameters import Parameters

# 读取配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

parser = argparse.ArgumentParser()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parameters = Parameters(parser, config['hardcoded_params'])  # 注入命令行参数和硬编码参数
    tracker = utils.Tracker(parameters, ['erl'], '_score.csv')  # 初始化跟踪器
    frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # 初始化跟踪器
    time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')
    ddpg_tracker = utils.Tracker(parameters, ['ddpg'], '_score.csv')
    selection_tracker = utils.Tracker(parameters, ['elite', 'selected', 'discarded'], '_selection.csv')

    # Create Env
    env = utils.NormalizedActions(gym.make(parameters.env_name))
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    parameters.write_params(stdout=True)

    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # Create Agent
    pderl = agents.PDERL(parameters, env)
    print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

    next_save = parameters.next_save; time_start = time.time()
    while pderl.num_frames <= parameters.num_frames:
        stats = pderl.train()
        best_train_fitness = stats['best_train_fitness']
        erl_score = stats['test_score']
        elite_index = stats['elite_index']
        ddpg_reward = stats['ddpg_reward']
        policy_gradient_loss = stats['pg_loss']
        behaviour_cloning_loss = stats['bc_loss']
        population_novelty = stats['pop_novelty']

        print('#Games:', pderl.num_games, '#Frames:', pderl.num_frames,
              ' Train_Max:', '%.2f'%best_train_fitness if best_train_fitness is not None else None,
              ' Test_Score:','%.2f'%erl_score if erl_score is not None else None,
              ' Avg:','%.2f'%tracker.all_tracker[0][1],
              ' ENV:  '+ parameters.env_name,
              ' DDPG Reward:', '%.2f'%ddpg_reward,
              ' PG Loss:', '%.4f' % policy_gradient_loss)

        elite = pderl.evolver.selection_stats['elite']/pderl.evolver.selection_stats['total']
        selected = pderl.evolver.selection_stats['selected'] / pderl.evolver.selection_stats['total']
        discarded = pderl.evolver.selection_stats['discarded'] / pderl.evolver.selection_stats['total']

        print()
        tracker.update([erl_score], pderl.num_games)
        frame_tracker.update([erl_score], pderl.num_frames)
        time_tracker.update([erl_score], time.time()-time_start)
        ddpg_tracker.update([ddpg_reward], pderl.num_frames)
        selection_tracker.update([elite, selected, discarded], pderl.num_frames)

        # Save Policy
        if pderl.num_games > next_save:
            next_save += parameters.next_save
            if elite_index is not None:
                torch.save(pderl.pop[elite_index].actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                   'evo_net.pkl'))

                if parameters.save_periodic:
                    save_folder = os.path.join(parameters.save_foldername, 'models')
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    actor_save_name = os.path.join(save_folder, 'evo_net_actor_{}.pkl'.format(next_save))
                    critic_save_name = os.path.join(save_folder, 'evo_net_critic_{}.pkl'.format(next_save))
                    buffer_save_name = os.path.join(save_folder, 'champion_buffer_{}.pkl'.format(next_save))

                    torch.save(pderl.pop[elite_index].actor.state_dict(), actor_save_name)
                    torch.save(pderl.rl_agent.critic.state_dict(), critic_save_name)
                    with open(buffer_save_name, 'wb+') as buffer_file:
                        pickle.dump(pderl.rl_agent.buffer, buffer_file)

            print("Progress Saved")











