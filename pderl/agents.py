import numpy as np
from scipy.spatial import distance
import torch
import torch.nn as nn
from torch.optim import Adam

from pderl.utils import is_lnorm_key
from pderl.parameters import Parameters
from pderl.replay_memory import ReplayMemory

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

    def size(self):
        return len(self.bcs)


class GeneticAgent:
    def __init__(self, args: Parameters):

        self.args = args

        self.actor = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-3)
        self.buffer = ReplayMemory(self.args.individual_bs, args.device)
        self.loss = nn.MSELoss()

    def update_parameters(self, batch, p1, p2, critic):
        state_batch, _, _, _, _ = batch

        p1_action = p1(state_batch)
        p2_action = p2(state_batch)
        p1_q = critic(state_batch, p1_action).flatten()
        p2_q = critic(state_batch, p2_action).flatten()

        eps = 0.0
        action_batch = torch.cat((p1_action[p1_q - p2_q > eps], p2_action[p2_q - p1_q >= eps])).detach()
        state_batch = torch.cat((state_batch[p1_q - p2_q > eps], state_batch[p2_q - p1_q >= eps]))
        actor_action = self.actor(state_batch)

        # Actor Update
        self.actor_optim.zero_grad()
        sq = (actor_action - action_batch)**2
        policy_loss = torch.sum(sq) + torch.mean(actor_action**2)
        policy_mse = torch.mean(sq)
        policy_loss.backward()
        self.actor_optim.step()

        return policy_mse.item()


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(input_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim),
        )
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        x = self.ln1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        return x


class Actor(nn.Module):
    def __init__(self, args, init=False):
        super(Actor, self).__init__()
        self.args = args
        self.embedding = nn.Linear(args.state_dim, args.actor_transformer_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args.actor_transformer_dim, args.actor_num_heads, args.actor_ff_dim, args.actor_dropout) for _ in range(args.actor_num_layers)]
        )
        self.fc = nn.Linear(args.actor_transformer_dim, args.action_dim)

        if init:
            self.fc.weight.data.mul_(0.1)
            self.fc.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input):
        x = self.embedding(input)
        x = x.unsqueeze(1)  # Add sequence dimension
        for block in self.transformer_blocks:
            x = block(x)
        x = x.squeeze(1)  # Remove sequence dimension
        out = torch.tanh(self.fc(x))
        return out

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state).cpu().data.numpy().flatten()

    def get_novelty(self, batch):
        state_batch, action_batch, _, _, _ = batch
        novelty = torch.mean(torch.sum((action_batch - self.forward(state_batch))**2, dim=-1))
        return novelty.item()

    # function to return current pytorch gradient in same order as genome's flattened parameter vector
    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.grad.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count + sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz

    # count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            count += param.numel()
        return count


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.state_embedding = nn.Linear(args.state_dim, args.critic_transformer_dim)
        self.action_embedding = nn.Linear(args.action_dim, args.critic_transformer_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args.critic_transformer_dim * 2, args.critic_num_heads, args.critic_ff_dim, args.critic_dropout) for _ in range(args.critic_num_layers)]
        )
        self.fc = nn.Linear(args.critic_transformer_dim * 2, 1)
        self.fc.weight.data.mul_(0.1)
        self.fc.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input, action):
        state_embed = self.state_embedding(input)
        action_embed = self.action_embedding(action)
        x = torch.cat([state_embed, action_embed], dim=-1)
        x = x.unsqueeze(1)  # Add sequence dimension
        for block in self.transformer_blocks:
            x = block(x)
        x = x.squeeze(1)  # Remove sequence dimension
        out = self.fc(x)
        return out