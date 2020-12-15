import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

from tictactoe import TicTacToe

class DQN(nn.Module):
    def __init__(self, n_states, n_hidden=10, n_conv=2, kernel_size=3):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, n_conv, kernel_size=kernel_size, padding=1),
            torch.nn.Flatten(), 
            nn.Linear(n_states * n_conv, n_hidden), 
            nn.ReLU(),
            nn.Linear(n_hidden, n_states)
        )

    def forward(self, x):
        q = self.net(x)
        q[x.view(q.shape) != 0] = -np.infty
        return q

class DuelingDQN(nn.Module):
    def __init__(self, n_states, n_hidden=10, n_conv=2, kernel_size=3):
        super(DuelingDQN, self).__init__()

        self.features_net = nn.Sequential(
            nn.Conv2d(1, n_conv, kernel_size=kernel_size, padding=1),
            torch.nn.Flatten(), 
            nn.Linear(n_states * n_conv, n_hidden), 
            nn.ReLU(),
        )
        
        self.values_net = nn.Sequential(
            nn.Linear(n_hidden, 1)
        )
        
        self.advantage_net = nn.Sequential(
            nn.Linear(n_hidden, n_states)
        )

    def forward(self, x):
        features = self.features_net(x)
        values = self.values_net(features)
        advantage = self.advantage_net(features)

        q = values + (advantage - advantage.mean())
        q[x.view(q.shape) != 0] = -np.infty
        return q

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, memory):
        '''
        memory contents: (s, a, r, s_prime)
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = memory
        self.position = (self.position + 1) % self.capacity
       
    def sample(self, batch_size):
        memory_batch = random.sample(self.memory, batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = zip(*memory_batch)

        s_batch = torch.cat(s_batch)
        a_batch = torch.tensor(a_batch)[:, None]
        r_batch = torch.tensor(r_batch, dtype=torch.float32)
        s_prime_batch = torch.cat(s_prime_batch) 

        return s_batch, a_batch, r_batch, s_prime_batch
    
    def __len__(self):
        return len(self.memory)
