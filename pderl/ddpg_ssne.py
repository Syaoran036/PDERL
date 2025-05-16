import torch
import torch.nn as nn
from torch.optim import Adam
from pderl import replay_memory
from parameters import Parameters
from pderl.utils import is_lnorm_key, hard_update, soft_update
import numpy as np
import fastrand
import random
from typing import List


class DDPG(object):
    def __init__(self, args):
        self.args = args
        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

        from agent_related import Actor, Critic
        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def td_error(self, state, action, next_state, reward, done):
        next_action = self.actor_target.forward(next_state)
        next_q = self.critic_target(next_state, next_action)

        done = 1 if done else 0
        if self.args.use_done_mask: next_q = next_q * (1 - done)  # Done mask
        target_q = reward + (self.gamma * next_q)

        current_q = self.critic(state, action)
        dt = (current_q - target_q).abs()
        return dt.item()

    def update_parameters(self, batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        # Load everything to GPU if not already
        self.actor_target.to(self.args.device)
        self.critic_target.to(self.args.device)
        self.critic.to(self.args.device)
        state_batch = state_batch.to(self.args.device)
        next_state_batch = next_state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_batch = reward_batch.to(self.args.device)
        if self.args.use_done_mask: done_batch = done_batch.to(self.args.device)

        # Critic Update
        next_action_batch = self.actor_target.forward(next_state_batch)
        next_q = self.critic_target.forward(next_state_batch, next_action_batch)
        if self.args.use_done_mask: next_q = next_q * (1 - done_batch) #Done mask
        target_q = reward_batch + (self.gamma * next_q).detach()

        self.critic_optim.zero_grad()
        current_q = self.critic.forward(state_batch, action_batch)
        delta = (current_q - target_q).abs()
        dt = torch.mean(delta**2)
        dt.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        # Actor Update
        self.actor_optim.zero_grad()

        policy_grad_loss = -(self.critic.forward(state_batch, self.actor.forward(state_batch))).mean()
        policy_loss = policy_grad_loss

        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_grad_loss.data.cpu().numpy(), delta.data.cpu().numpy()


class SSNE:
    def __init__(self, args: Parameters, critic, evaluate):
        self.current_gen = 0
        self.args = args;
        self.critic = critic
        self.population_size = self.args.pop_size
        self.num_elitists = int(self.args.elite_fraction * args.pop_size)
        self.evaluate = evaluate
        from pderl.utils import PopulationStats
        self.stats = PopulationStats(self.args)
        if self.num_elitists < 1: self.num_elitists = 1

        self.rl_policy = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded':0, 'total':0.0000001}

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def crossover_inplace(self, gene1, gene2):
        # Evaluate the parents
        trials = 5
        if self.args.opstat and self.stats.should_log():
            test_score_p1 = 0
            for eval in range(trials):
                episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p1 += episode['reward']
            test_score_p1 /= trials

            test_score_p2 = 0
            for eval in range(trials):
                episode = self.evaluate(gene2, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p2 += episode['reward']
            test_score_p2 /= trials

        for param1, param2 in zip(gene1.actor.parameters(), gene2.actor.parameters()):
            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2: #Weights no bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W2[ind_cr, :] = W1[ind_cr, :]

            elif len(W1.shape) == 1: #Bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = fastrand.pcg32bounded(num_variables)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W1[ind_cr] = W2[ind_cr]
                    else:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W2[ind_cr] = W1[ind_cr]

        # Evaluate the children
        if self.args.opstat and self.stats.should_log():
            test_score_c1 = 0
            for eval in range(trials):
                episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c1 += episode['reward']
            test_score_c1 /= trials

            test_score_c2 = 0
            for eval in range(trials):
                episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c2 += episode['reward']
            test_score_c2 /= trials

            if self.args.verbose_crossover:
                print("==================== Classic Crossover ======================")
                print("Parent 1", test_score_p1)
                print("Parent 2", test_score_p2)
                print("Child 1", test_score_c1)
                print("Child 2", test_score_c2)

            self.stats.add({
                'cros_parent1_fit': test_score_p1,
                'cros_parent2_fit': test_score_p2,
                'cros_child_fit': np.mean([test_score_c1, test_score_c2]),
                'cros_child1_fit': test_score_c1,
                'cros_child2_fit': test_score_c2,
            })
    
    def distilation_crossover(self, gene1, gene2):
        from agent_related import GeneticAgent
        new_agent = GeneticAgent(self.args)
        new_agent.buffer.add_latest_from(gene1.buffer, self.args.individual_bs // 2)
        new_agent.buffer.add_latest_from(gene2.buffer, self.args.individual_bs // 2)
        new_agent.buffer.shuffle()

        hard_update(new_agent.actor, gene2.actor)
        batch_size = min(128, len(new_agent.buffer))
        iters = len(new_agent.buffer) // batch_size
        losses = []
        for epoch in range(12):
            for i in range(iters):
                batch = new_agent.buffer.sample(batch_size)
                losses.append(new_agent.update_parameters(batch, gene1.actor, gene2.actor, self.critic))

        if self.args.opstat and self.stats.should_log():

            test_score_p1 = 0
            trials = 5
            for eval in range(trials):
                episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p1 += episode['reward']
            test_score_p1 /= trials

            test_score_p2 = 0
            for eval in range(trials):
                episode = self.evaluate(gene2, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p2 += episode['reward']
            test_score_p2 /= trials

            test_score_c = 0
            for eval in range(trials):
                episode = self.evaluate(new_agent, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c += episode['reward']
            test_score_c /= trials

            if self.args.verbose_crossover:
                print("==================== Distillation Crossover ======================")
                print("MSE Loss:", np.mean(losses[-40:]))
                print("Parent 1", test_score_p1)
                print("Parent 2", test_score_p2)
                print("Crossover performance: ", test_score_c)

            self.stats.add({
                'cros_parent1_fit': test_score_p1,
                'cros_parent2_fit': test_score_p2,
                'cros_child_fit': test_score_c,
            })

        return new_agent

    def mutate_inplace(self, gene):
        trials = 5
        if self.stats.should_log():
            test_score_p = 0
            for eval in range(trials):
                episode = self.evaluate(gene, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p += episode['reward']
            test_score_p /= trials

        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        num_params = len(list(gene.actor.parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
        model_params = gene.actor.state_dict()

        for i, key in enumerate(model_params): #Mutate each param

            if is_lnorm_key(key):
                continue

            # References to the variable keys
            W = model_params[key]
            # ... 剩余未显示代码 ...

    def epoch(self, pop, all_fitness):
        # 假设这里有 epoch 方法的实现
        pass