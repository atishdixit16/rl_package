#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
import os

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import OrderedDict

from rl_package.utils.set_seed import set_seed
from rl_package.utils.multiprocessing_env import SubprocVecEnv


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def get_activation(activation):
    if activation=='relu':
        return nn.ReLU()
    elif activation=='tanh':
        return nn.Tanh()
    elif activation=='sigmoid':
        return nn.Sigmoid()
    else:
        raise Exception('invalid activation key. should be one of these: relu, tanh or sigmoid')

def get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, network_key):
    module_list = {}
    for layer, activation, i in zip( MLP_LAYERS, MLP_ACTIVATIONS, range(len(MLP_LAYERS)) ):
        if i==0:
            module_list['layer '+str(i)] = nn.Linear(num_inputs, layer)
            module_list['layer '+str(i)+' act'] = get_activation(activation)
            last_layer = layer
        else:
            module_list['layer '+str(i)] = nn.Linear(last_layer, layer)
            module_list['layer '+str(i)+' act'] = get_activation(activation)
            last_layer = layer
    if network_key=='actor':
        module_list['layer '+str(i+1)] = nn.Linear(last_layer, num_outputs)
    elif network_key=='critic':
        module_list['layer '+str(i+1)] = nn.Linear(last_layer, 1)
    else:
        raise Exception('invalid network key. should be one of these: actor or critic')
    return module_list


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential ( OrderedDict (get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS,'critic') )  )
        self.actor = nn.Sequential ( OrderedDict (get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS,'actor') ) )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

def test_env(env, model, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def a2c_algorithm(ENV, NUM_ENV=8,
                  TOTAL_STEPS=400000, NSTEPS=32,
                  VF_COEF=0.5, ENT_COEF=0.001,
                  GAMMA=0.95,
                  MLP_LAYERS=[64, 64], MLP_ACTIVATIONS=['relu', 'relu'], LEARNING_RATE=1e-4,
                  PRINT_FREQ=8000, N_TEST_ENV=50, 
                  SAVE_RESULTS=False, FILE_PATH='results/', LOG_FILE_NAME='log', SAVE_MODEL=False, MODEL_FILE_NAME='model',
                  SEED=1):

    # reproducibility
    set_seed(SEED)

    assert not PRINT_FREQ % NUM_ENV, 'Invalid print frequency. For convinience, select such that PRINT_FREQ % NUM_ENV = 0'

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    num_envs = NUM_ENV
    env = ENV    
    env.seed(SEED)

    def make_env():
        def _thunk():
            env.seed(SEED)
            return env
        return _thunk

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.shape[0]

    #Hyper params:
    lr               = LEARNING_RATE
    num_steps        = NSTEPS

    model = ActorCritic(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_steps = TOTAL_STEPS
    steps  = 0
    timesteps = []
    test_rewards = []
    c_loss_array = []
    c_loss = []

    state = envs.reset()

    while steps < total_steps :

        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy   = 0

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            state = next_state
            steps += num_envs

            if not steps % PRINT_FREQ:
                test_reward = np.mean([test_env(env, model) for _ in range(N_TEST_ENV)])
                test_rewards.append(test_reward)
                timesteps.append(steps)
                c_loss_array.append(np.mean(c_loss))
                print('timestep : {}, reward: {}, c_loss: {}'.format(steps, round(test_reward), round(np.mean(c_loss)) ))
                c_loss = []

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks, GAMMA)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        advantage = returns - values

        actor_loss  = -(log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + VF_COEF * critic_loss - ENT_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        clipping_value = 10.0 # arbitrary value of your choosing
        nn.utils.clip_grad_value_(model.parameters(), clipping_value)
        optimizer.step()

        c_loss.append(critic_loss)
        

    if SAVE_RESULTS:
        output_table = np.stack((timesteps, test_rewards, c_loss_array))
        if not os.path.exists(FILE_PATH):
            os.makedirs(FILE_PATH)
        file_name = FILE_PATH+LOG_FILE_NAME+'.csv'
        np.savetxt(file_name, np.transpose(output_table), delimiter=',', header='Timestep,Rewards,Critic_loss')
        if SAVE_MODEL:
            torch.save(model.state_dict(), FILE_PATH+MODEL_FILE_NAME )
        return model

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    model = a2c_algorithm(env)