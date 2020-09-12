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

def get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, network_key):
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
    
    if network_key=='actor':
        if ACTOR_FINAL_ACTIVATION is not None:
            module_list['final layer act']=get_activation(ACTOR_FINAL_ACTIVATION)

    return module_list


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential ( OrderedDict (get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, 'critic') )  )
        self.actor = nn.Sequential ( OrderedDict (get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, 'actor') ) )
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


def compute_gae(next_value, rewards, masks, values, GAMMA, LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
        gae = delta + GAMMA * LAMBDA * masks[step] * gae
        returns.insert(0, gae+values[step]) # originally returns.insert(0, gae + values[step]), not sure why we need values[step]!
    return returns

        
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids[:batch_size // mini_batch_size * mini_batch_size], batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, model, optimizer, CLIP_PARAM, VF_COEF, ENT_COEF  ):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = VF_COEF * critic_loss + actor_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # print(loss)


def ppo_algorithm(ENV, NUM_ENV=8,
                  TOTAL_STEPS=240000, NSTEPS=64, MINIBATCH_SIZE=128, N_EPOCH=30,
                  CLIP_PARAM=0.1, VF_COEF=0.5, ENT_COEF=0.001,
                  GAMMA=0.99, LAMBDA=0.95,
                  MLP_LAYERS=[64,64], MLP_ACTIVATIONS=['relu', 'relu'], ACTOR_FINAL_ACTIVATION='None', ACTOR_DIST_LOG_STD=0.0, LEARNING_RATE=1e-3,
                  PRINT_FREQ=8000, N_TEST_ENV=50, TEST_ENV_FUNC=test_env,
                  SAVE_RESULTS=False, FILE_PATH='results/', LOG_FILE_NAME='log', SAVE_MODEL=False, MODEL_FILE_NAME='model',
                  SEED=4):

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
    mini_batch_size  = MINIBATCH_SIZE
    ppo_epochs       = N_EPOCH

    model = ActorCritic(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, std=ACTOR_DIST_LOG_STD).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_steps = TOTAL_STEPS
    steps  = 0
    timesteps = []
    test_rewards = []

    state = envs.reset()

    while steps < total_steps :

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            states.append(state)
            actions.append(action)

            state = next_state
            if done.any():
                state[done] = envs.reset()[done]
            steps += num_envs


            if not steps % PRINT_FREQ:
                test_reward = np.mean([TEST_ENV_FUNC(env, model) for _ in range(N_TEST_ENV)])
                test_rewards.append(test_reward)
                timesteps.append(steps)
                print('timestep : {}, reward: {}'.format(steps, round(test_reward)))

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values, GAMMA, LAMBDA)


        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values

        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, model, optimizer, CLIP_PARAM, VF_COEF, ENT_COEF)

    if SAVE_RESULTS:
        output_table = np.stack((timesteps, test_rewards))
        if not os.path.exists(FILE_PATH):
            os.makedirs(FILE_PATH)
        file_name = FILE_PATH+LOG_FILE_NAME+'.csv'
        np.savetxt(file_name, np.transpose(output_table), delimiter=',', header='Timestep,Rewards')
        if SAVE_MODEL:
            torch.save(model.state_dict(), FILE_PATH+MODEL_FILE_NAME )
        return model

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    model = ppo_algorithm(env)