import copy
import collections
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .misc import to_var, clip_grads


class DoubleDQN:
    def __init__(self, model, action_n,
                 gamma=0.9, memory_size=10000,
                 learning_rate=1.e-4, batch_size=64,
                 episode2thresh=lambda i: 0.05 + 0.9 * np.exp(-1. * i / 100)):
        self.model = model  # actor model
        self.target_model = copy.deepcopy(model)
        self.memory = collections.deque(maxlen=memory_size)
        self.gamma = gamma
        self.action_n = action_n
        self.batch_size = batch_size
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.episode2thresh = episode2thresh

        self.target_model.eval()  # target_model will always be eval mode

        # self.history = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, obs, episode=np.inf):
        thresh = self.episode2thresh(episode)
        if np.random.random() < thresh:
            action = np.random.randint(self.action_n)
        else:
            state = to_var(torch.Tensor(obs).unsqueeze(0))
            q_values = self._get_q_value(state)
            _, action_ = q_values.max(1)
            action = action_.data[0]
        return action

    def _get_q_value(self, state):
        self.model.eval()
        with torch.no_grad():
            values = self.model(state)
        return values

    def _get_target_q_value(self, state):
        with torch.no_grad():
            values = self.target_model(state)
        return values

    def memorize(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def _replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        state_batch = to_var(torch.stack([torch.Tensor(b[0]) for b in batch]))

        action_batch = to_var(torch.stack([torch.LongTensor([b[1]])
                                           for b in batch]))

        next_state_batch = to_var(torch.stack([torch.Tensor(b[2])
                                               for b in batch if b[2] is not None]))
        non_final_mask = torch.ByteTensor([b[2] is not None for b in batch])

        reward_batch = to_var(torch.stack([torch.Tensor([b[3]])
                                           for b in batch]))

        self.model.train()
        curr_values = self.model(state_batch).gather(1, action_batch)

        next_state_q = self._get_q_value(next_state_batch)
        next_action_batch = next_state_q.max(1)[1].unsqueeze(-1)
        next_state_target_q = self._get_target_q_value(next_state_batch)
        next_values = to_var(torch.zeros(batch_size, 1).float())
        next_values[non_final_mask] = next_state_target_q\
            .gather(1, next_action_batch)

        expected_values = next_values * self.gamma + reward_batch

        loss = self.loss_fn(curr_values, expected_values)

        # self.history.append([loss.data[0]])  # train_loss
        return loss

    def step(self):
        if len(self.memory) > self.batch_size:
            loss = self._replay(self.batch_size)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grads(self.model, -5, 5)
            self.optimizer.step()
            return loss.data[0]
        else:
            # print("Not enough experience.")
            pass

    def play(self, obs):
        state = to_var(torch.Tensor(obs).unsqueeze(0))
        q_values = self._get_q_value(state)
        _, action_ = q_values.max(1)
        action = action_.data[0]
        return action
