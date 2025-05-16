import numpy as np, os, time, random
from pderl import utils, agents
import gymnasium as gym, torch
import pickle
import json

from pderl.parameters import Parameters
from pderl.agents import GeneticAgent
from pderl.ddpg_ssne import DDPG, OUNoise, SSNE
from pderl.replay_memory import ReplayMemory, PrioritizedReplayMemory, Transition

class PDERL:
    def __init__(self, args: Parameters, env):
        self.args = args; self.env = env

        # Init population
        self.pop = []
        self.buffers = []
        for _ in range(args.pop_size):
            self.pop.append(GeneticAgent(args))

        # Init RL Agent
        self.rl_agent = DDPG(args)
        self.ounoise = OUNoise(args.action_dim)
        self.evolver = SSNE(self.args, self.rl_agent.critic, self.evaluate)
        
        if args.per:
            self.replay_buffer = PrioritizedReplayMemory(args.buffer_size, args.device,
                                                                       beta_frames=self.args.num_frames)
        else:
            self.replay_buffer = ReplayMemory(args.buffer_size, args.device)

        # Population novelty
        self.ns_r = 1.0
        self.ns_delta = 0.1
        self.best_train_reward = 0.0
        self.time_since_improv = 0
        self.step = 1

        # Trackers
        self.num_games = 0; self.num_frames = 0; self.iterations = 0; self.gen_frames = None

    def evaluate(self, agent, is_render=False, is_action_noise=False,
                 store_transition=True, net_index=None):
        total_reward = 0.0
        total_error = 0.0

        state,_ = self.env.reset()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            if self.args.render and is_render: self.env.render()
            action = agent.actor.select_action(np.array(state))
            if is_action_noise:
                action += self.ounoise.noise()
                action = np.clip(action, -1.0, 1.0)

            # Simulate one step in environment
            next_state, reward, done, _, info = self.env.step(action.flatten())
            #observation, reward, terminated, truncated, info
            total_reward += reward

            transition = (state, action, next_state, reward, float(done))
            if store_transition:
                self.replay_buffer.add(*transition)
                agent.buffer.add(*transition)

            state = next_state
        if store_transition: self.num_games += 1

        return {'reward': total_reward, 'td_error': total_error}

    def rl_to_evo(self, rl_agent, evo_net):
        for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
            target_param.data.copy_(param.data)
        evo_net.buffer.reset()
        evo_net.buffer.add_content_of(rl_agent.buffer)

    def evo_to_rl(self, rl_net, evo_net):
        for target_param, param in zip(rl_net.parameters(), evo_net.parameters()):
            target_param.data.copy_(param.data)

    def get_pop_novelty(self):
        epochs = self.args.ns_epochs
        novelties = np.zeros(len(self.pop))
        for _ in range(epochs):
            transitions = self.replay_buffer.sample(self.args.batch_size)
            batch = Transition(*zip(*transitions))

            for i, net in enumerate(self.pop):
                novelties[i] += (net.get_novelty(batch))
        return novelties / epochs

    def train_ddpg(self):
        bcs_loss, pgs_loss = [], []
        if len(self.replay_buffer) > self.args.batch_size * 5:
            for _ in range(int(self.gen_frames * self.args.frac_frames_train)):
                batch = self.replay_buffer.sample(self.args.batch_size)

                pgl, delta = self.rl_agent.update_parameters(batch)
                pgs_loss.append(pgl)

        return {'bcs_loss': 0, 'pgs_loss': pgs_loss}

    def train(self):
        self.gen_frames = 0
        self.iterations += 1

        # ========================== EVOLUTION  ==========================
        # Evaluate genomes/individuals
        rewards = np.zeros(len(self.pop))
        errors = np.zeros(len(self.pop))
        for i, net in enumerate(self.pop):
            for _ in range(self.args.num_evals):
                episode = self.evaluate(net, is_render=False, is_action_noise=False, net_index=i)
                rewards[i] += episode['reward']
                errors[i] += episode['td_error']

        rewards /= self.args.num_evals
        errors /= self.args.num_evals

        # all_fitness = 0.8 * rankdata(rewards) + 0.2 * rankdata(errors)
        all_fitness = rewards

        # Validation test for NeuroEvolution champion
        best_train_fitness = np.max(rewards)
        champion = self.pop[np.argmax(rewards)]

        # print("Best TD Error:", np.max(errors))

        test_score = 0
        for eval in range(5):
            episode = self.evaluate(champion, is_render=True, is_action_noise=False, store_transition=False)
            test_score += episode['reward']
        test_score /= 5.0

        # NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, all_fitness)

        # ========================== DDPG ===========================
        # Collect experience for training
        self.evaluate(self.rl_agent, is_action_noise=True)

        losses = self.train_ddpg()

        # Validation test for RL agent
        testr = 0
        for eval in range(5):
            ddpg_stats = self.evaluate(self.rl_agent, store_transition=False, is_action_noise=False)
            testr += ddpg_stats['reward']
        testr /= 5

        # Sync RL Agent to NE every few steps
        if self.iterations % self.args.rl_to_ea_synch_period == 0:
            # Replace any index different from the new elite
            replace_index = np.argmin(all_fitness)
            if replace_index == elite_index:
                replace_index = (replace_index + 1) % len(self.pop)

            self.rl_to_evo(self.rl_agent, self.pop[replace_index])
            self.evolver.rl_policy = replace_index
            print('Sync from RL --> Nevo')

        # -------------------------- Collect statistics --------------------------
        return {
            'best_train_fitness': best_train_fitness,
            'test_score': test_score,
            'elite_index': elite_index,
            'ddpg_reward': testr,
            'pg_loss': np.mean(losses['pgs_loss']),
            'bc_loss': np.mean(losses['bcs_loss']),
            'pop_novelty': np.mean(0),
        }


# 读取配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parameters = Parameters(config)  # 注入命令行参数和硬编码参数
    tracker = utils.Tracker(parameters, ['erl'], '_score.csv')  # 初始化跟踪器
    frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # 初始化跟踪器
    time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')
    ddpg_tracker = utils.Tracker(parameters, ['ddpg'], '_score.csv')
    selection_tracker = utils.Tracker(parameters, ['elite', 'selected', 'discarded'], '_selection.csv')

    # Create Env
    env = utils.NormalizedActions(gym.make(parameters.env))
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    parameters.write_params(stdout=True)

    # Seed
    # env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # Create Agent
    pderl = PDERL(parameters, env)
    print('Running', parameters.env, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

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
              ' ENV:  '+ parameters.env,
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











