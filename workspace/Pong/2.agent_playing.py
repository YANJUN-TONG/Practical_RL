
# coding: utf-8

# In[1]:

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:

import gym
import torch


# In[3]:

import _init_paths


# In[4]:

from rllib.models import ConvNet
from rllib.reinforce import REINFORCE


# In[5]:

downsample = 2
output_size = 160//downsample

def preprocess(frame):
    '''from karpathy.'''
    I = frame
    I = I[35:195] # crop
    I = I[::downsample,::downsample,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    tensor = torch.from_numpy(I).float()
    return tensor.unsqueeze(0) #BCHW


# In[6]:

env = gym.make("Pong-v0")

net = ConvNet(input_shape=(1,output_size,output_size), action_n=env.action_space.n)
weights_path = "reinforce.pth"

agent = REINFORCE(model=net, gamma=0.99, learning_rate=1.e-3, batch_size=10)

if torch.cuda.is_available():
    net = net.cuda()
    weights = torch.load(weights_path)
else:
    weights = torch.load(weights_path, map_location={'cuda:0': 'cpu'})
net.load_state_dict(weights)

print(net)


# In[7]:

for episode in range(10):
    frame = env.reset()
    last_obs = preprocess(frame)
    curr_obs = preprocess(frame)
    total_reward = 0
    for step in range(100000): # not exceed 10000 steps
        action = agent.play(curr_obs-last_obs)
        frame, reward, done, _ = env.step(action)
        env.render()
        last_obs = curr_obs
        curr_obs = preprocess(frame)
        total_reward+=reward
        if done:
             break
    print(episode, total_reward)


# In[ ]:



