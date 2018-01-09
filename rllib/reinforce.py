import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical

from .misc import clip_grads, to_var


class REINFORCE:
    '''Implement REINFORCE algorithm.'''

    def __init__(self, model, gamma=0.99, learning_rate=1.e-3, batch_size=10):
        self.model = model
        self.gamma = gamma
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()  # need or not?
        self.batch_size = batch_size

        self.log_probs = []
        self.rewards = []

        self.history = []

    @property
    def episode(self):
        return len(self.history)

    def select_action(self, obs):
        self.model.train()
        state = to_var(obs)
        logits = self.model(state)
        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action, log_prob

    def keep_for_policy_grad(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def _accumulate_policy_grad(self):
        policy_loss = get_policy_loss(self.log_probs, self.rewards, self.gamma)

        self.history.append([sum(self.rewards),  # total_reward
                             len(self.rewards),  # n_round
                             policy_loss.data[0]])  # train_loss

        policy_loss.backward()
        del self.log_probs[:]
        del self.rewards[:]

    def _train(self):
        clip_grads(self.model, -10, 10)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self):
        self._accumulate_policy_grad()
        episode = self.episode
        if episode > 0 and episode % self.batch_size == 0:
            self._train()

    def play(self, obs):
        self.model.eval()
        state = to_var(obs)
        prob = self.model(state)
        _, action = prob.max(dim=1)
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


def get_policy_loss(log_probs, rewards, gamma):
    ret = 0
    normalized_rewards = get_normalized_rewards(rewards, gamma)
    for log_prob, reward in zip(log_probs, normalized_rewards):
        ret -= log_prob * reward  # it's less memory consuming than dot product
    return ret
