import numpy as np, os, time, random
from pderl import utils as utils
import gymnasium as gym, torch
import argparse
import pickle
import json
from pderl.operator_runner import OperatorRunner
from pderl.parameters import Parameters
from pderl import replay_memory
from torch.autograd import Variable
import fastrand, math
import torch.distributions as dist
from pderl.ddpg import hard_update
from typing import List
from scipy.spatial import distance
from scipy.stats import rankdata


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Tracker:
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = parameters.save_foldername
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] # [Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        self.counter = 0
        self.conv_size = 10
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        self.counter += 1
        for update, var in zip(updates, self.all_tracker):
            if update == None: continue
            var[0].append(update)

        # Constrain size of convolution
        for var in self.all_tracker:
            if len(var[0]) > self.conv_size: var[0].pop(0)

        # Update new average
        for var in self.all_tracker:
            if len(var[0]) == 0: continue
            var[1] = sum(var[0])/float(len(var[0]))

        if self.counter % 4 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0: continue
                var[2].append(np.array([generation, var[1]]))
                filename = os.path.join(self.foldername, self.vars_string[i] + self.project_string)
                try:
                    np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')
                except:
                    # Common error showing up in the cluster for unknown reasons
                    print('Failed to save progress')


class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    #v = 1. / np.sqrt(fanin)
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)


def to_numpy(var):
    return var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)


def pickle_obj(filename, object):
    handle = open(filename, "wb")
    pickle.dump(object, handle)


def unpickle_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def odict_to_numpy(odict):
    l = list(odict.values())
    state = l[0]
    for i in range(1, len(l)):
        if isinstance(l[i], np.ndarray):
            state = np.concatenate((state, l[i]))
        else: #Floats
            state = np.concatenate((state, np.array([l[i]])))
    return state


def min_max_normalize(x):
    min_x = np.min(x)
    max_x = np.max(x)
    return (x - min_x) / (max_x - min_x)


def is_lnorm_key(key):
    return key.startswith('lnorm')


def main():
    # 读取配置文件
    with open('config.json', 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    for arg_name, arg_config in config['cli_args'].items():
        if 'action' in arg_config:
            parser.add_argument(f'-{arg_name}', help=arg_config['help'], action=arg_config['action'])
        else:
            parser.add_argument(f'-{arg_name}', help=arg_config['help'], type=eval(arg_config['type']), default=arg_config.get('default'))

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    if __name__ == "__main__":
        parameters = Parameters(parser, config['hardcoded_params'])  # 注入命令行参数和硬编码参数
        tracker = Tracker(parameters, ['erl'], '_score.csv')  # 初始化跟踪器
        frame_tracker = Tracker(parameters, ['frame_erl'], '_score.csv')  # 初始化跟踪器
        time_tracker = Tracker(parameters, ['time_erl'], '_score.csv')
        ddpg_tracker = Tracker(parameters, ['ddpg'], '_score.csv')
        selection_tracker = Tracker(parameters, ['elite', 'selected', 'discarded'], '_selection.csv')

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

        # Tests the variation operators after that is saved first with -save_periodic
        # 移除 test_operators 相关逻辑
        # if parameters.test_operators:
        #     operator_runner = OperatorRunner(parameters, env)
        #     operator_runner.run()
        #     exit()

        # Create Agent
        from agent_related import Agent
        from ddpg_ssne import DDPG, SSNE
        agent = Agent(parameters, env)
        agent.rl_agent = DDPG(parameters)
        agent.ounoise = None  # 需根据实际情况补充
        agent.evolver = SSNE(parameters, agent.rl_agent.critic, agent.evaluate)

        print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

        next_save = parameters.next_save; time_start = time.time()
        while agent.num_frames <= parameters.num_frames:
            stats = agent.train()
            best_train_fitness = stats['best_train_fitness']
            erl_score = stats['test_score']
            elite_index = stats['elite_index']
            ddpg_reward = stats['ddpg_reward']
            policy_gradient_loss = stats['pg_loss']
            behaviour_cloning_loss = stats['bc_loss']
            population_novelty = stats['pop_novelty']

            print('#Games:', agent.num_games, '#Frames:', agent.num_frames,
                  ' Train_Max:', '%.2f'%best_train_fitness if best_train_fitness is not None else None,
                  ' Test_Score:','%.2f'%erl_score if erl_score is not None else None,
                  ' Avg:','%.2f'%tracker.all_tracker[0][1],
                  ' ENV:  '+ parameters.env_name,
                  ' DDPG Reward:', '%.2f'%ddpg_reward,
                  ' PG Loss:', '%.4f' % policy_gradient_loss)

            elite = agent.evolver.selection_stats['elite']/agent.evolver.selection_stats['total']
            selected = agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']
            discarded = agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']

            print()
            tracker.update([erl_score], agent.num_games)
            frame_tracker.update([erl_score], agent.num_frames)
            time_tracker.update([erl_score], time.time()-time_start)
            ddpg_tracker.update([ddpg_reward], agent.num_frames)
            selection_tracker.update([elite, selected, discarded], agent.num_frames)

            # Save Policy
            if agent.num_games > next_save:
                next_save += parameters.next_save
                if elite_index is not None:
                    torch.save(agent.pop[elite_index].actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                       'evo_net.pkl'))

                    if parameters.save_periodic:
                        save_folder = os.path.join(parameters.save_foldername, 'models')
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

                        actor_save_name = os.path.join(save_folder, 'evo_net_actor_{}.pkl'.format(next_save))
                        critic_save_name = os.path.join(save_folder, 'evo_net_critic_{}.pkl'.format(next_save))
                        buffer_save_name = os.path.join(save_folder, 'champion_buffer_{}.pkl'.format(next_save))

                        torch.save(agent.pop[elite_index].actor.state_dict(), actor_save_name)
                        torch.save(agent.rl_agent.critic.state_dict(), critic_save_name)
                        with open(buffer_save_name, 'wb+') as buffer_file:
                            pickle.dump(agent.rl_agent.buffer, buffer_file)

                print("Progress Saved")


class Archive:
    """A record of past behaviour characterisations (BC) in the population"""

    def __init__(self, args):
        self.args = args
        # Past behaviours
        self.bcs = []

    def add_bc(self, bc):
        if len(self.bcs) + 1 > self.args.archive_size:
            self.bcs = self.bcs[1:]
        self.bcs.append(bc)

    def get_novelty(self, this_bc):
        if self.size() == 0:
            return np.array(this_bc).T @ np.array(this_bc)
        distances = np.ravel(distance.cdist(np.expand_dims(this_bc, axis=0), np.array(self.bcs), metric='sqeuclidean'))
        distances = np.sort(distances)
        return distances[:self.args.ns_k].mean()


if __name__ == "__main__":
    main()