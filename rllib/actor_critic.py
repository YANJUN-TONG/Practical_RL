import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical

from .misc import clip_grads, to_var


class ActorCritic:
    '''Implement Actor-Critic algorithm.'''

    def __init__(self, model, gamma=0.99, learning_rate=1.e-3, batch_size=10):
        self.model = model
        self.gamma = gamma
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()  # need or not?
        self.batch_size = batch_size

        self.log_probs = []
        self.state_values = []
        self.rewards = []

        self.history = []

    @property
    def episode(self):
        return len(self.history)

    def select_action(self, obs):
        self.model.train()
        state = to_var(torch.Tensor(obs).unsqueeze(0))
        logits, state_value = self.model(state)
        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.data[0], log_prob, state_value

    def keep_for_grad(self, log_prob, state_value, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.state_values.append(state_value)

    def _accumulate_grad(self):
        policy_loss, value_loss = get_loss(self.log_probs, self.state_values,
                                           self.rewards, self.gamma)

        self.history.append([sum(self.rewards),  # total_reward
                             len(self.rewards),  # n_round
                             policy_loss.data[0],  # policy_loss
                             value_loss.data[0]])  # value_loss

        loss = policy_loss + value_loss

        loss.backward()

        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]

    def _train(self):
        clip_grads(self.model, -10, 10)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self):
        self._accumulate_grad()
        episode = self.episode
        if episode > 0 and episode % self.batch_size == 0:
            self._train()

    def play(self, obs):
        self.model.eval()
        state = to_var(torch.Tensor(obs).unsqueeze(0))
        with torch.no_grad():
            logits, _ = self.model(state)
        _, action = logits.max(dim=1)
        return action.data[0]


def get_discounted_rewards(rewards, gamma):
    acc = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        acc.append(R)
    ret = np.array(acc[::-1])
    return ret


def get_normalized_rewards(rewards, gamma):
    ret = get_discounted_rewards(rewards, gamma)
    return (ret - ret.mean()) / (ret.std() + np.finfo(np.float32).eps)


def get_loss(log_probs, state_values, rewards, gamma):
    policy_loss = 0
    value_loss = 0
    normalized_rewards = get_normalized_rewards(rewards, gamma)
    for log_prob, state_value, reward in zip(log_probs, state_values, normalized_rewards):
        # it's less memory consuming than dot product
        policy_loss -= log_prob * (reward - state_value.data[0, 0])
        value_loss += F.smooth_l1_loss(state_value,
                                       to_var(torch.Tensor([[reward]])))
    return policy_loss, value_loss
