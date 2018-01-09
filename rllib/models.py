import torch
import torch.nn as nn
from torch.autograd import Variable

from .misc import total_weights


class ConvNet(nn.Module):
    '''Simple ConvNet for discrete outputs.'''

    def __init__(self, input_shape, action_n):
        '''
        input_shape=(1, 80, 80) # CHW
        action_n=6 # number of action space
        '''
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                  nn.ReLU())
        flatten_size = get_flatten_size(self.conv, input_shape)
        self.fc = nn.Sequential(nn.Linear(flatten_size, 512), nn.ReLU(),
                                nn.Linear(512, action_n))
        self.apply(weights_init)
        print("Network size:", total_weights(self))

    def forward(self, x):
        feat = self.conv(x)
        # for policy network, this can be useful for annealing
        logit = self.fc(feat.view(feat.size(0), -1))
        return logit


class ConvNetPV(nn.Module):
    '''ConvNet with (discrete) policy head and value head.'''

    def __init__(self, input_shape, action_n):
        '''
        input_shape=(1, 80, 80) # CHW
        action_n=6 # number of action space
        '''
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                  nn.ReLU())
        flatten_size = get_flatten_size(self.conv, input_shape)
        self.fc = nn.Sequential(nn.Linear(flatten_size, 512), nn.ReLU())
        self.policy_head = nn.Linear(512, action_n)
        self.value_head = nn.Linear(512, 1)
        self.apply(weights_init)
        print("Network size:", total_weights(self))

    def forward(self, x):
        feat = self.conv(x)
        fc_feat = self.fc(feat.view(feat.size(0), -1))
        # for policy network, this can be useful for annealing
        logit = self.policy_head(fc_feat)
        value = self.value_head(fc_feat)
        return logit, value


def get_flatten_size(module, input_shape):
    x = Variable(torch.rand(1, *input_shape))
    output_feat = module(x)
    n_size = output_feat.view(-1).size(0)
    return n_size


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)
        print("Initialized", m)
