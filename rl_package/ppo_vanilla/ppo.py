#!/usr/bin/env python
# coding: utf-8

# In[1]:

import random
import numpy as np
import os


import torch
import torch.nn as nn
import torch.optim as optim

from rl_package.utils.set_seed import set_seed
from rl_package.utils.multiprocessing_env import SubprocVecEnv
from rl_package.utils.env_wrappers import ParallelEnvWrapper


def test_env(env, model, device, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = model(state.to(device))
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def test_env_mean_return(envs, model, device, n_trials):
    num_env = envs.nenvs
    envs = ParallelEnvWrapper(envs)
    mean_return = []
    for _ in range(int(n_trials/num_env)):
        state = envs.reset()
        done = [False]*num_env
        total_reward = [0]*num_env
        while not np.array(done).all():
            state = torch.FloatTensor(state).unsqueeze(0)
            dist, _ = model(state.to(device))
            actions = dist.sample().cpu().numpy()[0]
            state, reward, done, _ = envs.step(list(actions))
            total_reward += reward
        mean_return.append(np.mean(total_reward))
    return np.mean(mean_return)


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
        yield states[ids[i]], actions[ids[i]], log_probs[ids[i]], returns[ids[i]], advantage[ids[i]]        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, model, optimizer, scheduler, CLIP_PARAM, VF_COEF, ENT_COEF, GRAD_CLIP, LR_ANNEAL  ):
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
            if GRAD_CLIP:
                nn.utils.clip_grad_value_(model.parameters(), 5)
            loss.backward()
            optimizer.step()
            if LR_ANNEAL:
                scheduler.step()
    # print(loss)


def ppo_algorithm(ENV, MODEL,
                  NUM_ENV=8,
                  TOTAL_STEPS=200000, NSTEPS=64, MINIBATCH_SIZE=128, N_EPOCH=30,
                  CLIP_PARAM=0.1, VF_COEF=0.5, ENT_COEF=0.001,
                  GAMMA=0.99, LAMBDA=0.95,
                  LEARNING_RATE=1e-3, GRAD_CLIP=False, LR_ANNEAL=False, NN_INIT='normal',
                  PRINT_FREQ=8000, N_TEST_ENV=48, TEST_ENV_FUNC=test_env_mean_return,
                  SAVE_RESULTS=False, FILE_PATH='results/', LOG_FILE_NAME='log', SAVE_MODEL=False, MODEL_FILE_NAME='model',
                  SEED=4):

    '''
        PPO parameters:
        ENV : environment class object, 
        NUM_ENV : number of vectorized environments,
        TOTAL_STEPS : Total number of timesteps, 
        NSTEPS : Number of steps in each iteration (smaller than terminal step), 
        MINIBATCH_SIZE : size minibatch used to train the PPO network, 
        N_EPOCH : no. of epoch in network training,
        CLIP_PARAM : clipping parameter in PPO, 
        VF_COEF : values function coefficient, 
        ENT_COEF : entropy coefficient,
        GAMMA : dicount factor, 
        LAMBDA : lambda return term for genaralized advantage estimator,
        LEARNING_RATE : learning rate for Adam ,
        PRINT_FREQ : print frequeny for no. of steps, 
        N_TEST_ENV : number for test env samples for averaging, 
        TEST_ENV_FUNC : function to test environment with the model,
        SAVE_RESULTS : boolean to specify whether to save results, 
        FILE_PATH : file path to save reults, 
        LOG_FILE_NAME : log file name initial, 
        SAVE_MODEL : boolean variable to specify whether to save model, 
        MODEL_FILE_NAME : model file name initial,
        SEED : seed for reporoducibility
    '''
    # reproducibility
    set_seed(SEED)

    assert not PRINT_FREQ % NUM_ENV, 'Invalid print frequency. For convinience, select such that PRINT_FREQ % NUM_ENV = 0'
    assert not N_TEST_ENV % NUM_ENV, 'Invalid no of trials for test env. For convinience, select such that N_TEST_ENV % NUM_ENV = 0'

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

    #Hyper params:
    lr               = LEARNING_RATE
    num_steps        = NSTEPS
    mini_batch_size  = MINIBATCH_SIZE
    ppo_epochs       = N_EPOCH
    total_steps = TOTAL_STEPS
    steps  = 0
    timesteps = []
    test_rewards = []

    model = MODEL.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lam = lambda steps: 1-steps/total_steps
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)

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
            # action = action.reshape(-1,1) # reshape 1-d categorical samples (in case of discrete action space)
            actions.append(action)

            state = next_state
            if done.any():
                state[done] = envs.reset()[done]
            steps += num_envs


            if not steps % PRINT_FREQ:
                test_reward = TEST_ENV_FUNC(envs, model, device, n_trials=N_TEST_ENV)
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

        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, model, optimizer, scheduler, CLIP_PARAM, VF_COEF, ENT_COEF, GRAD_CLIP, LR_ANNEAL)

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
    # an example
    import gym
    from rl_package.utils.standard_nn_architectures import ActorCriticDense
    env = gym.make('Pendulum-v0')
    model = ActorCriticDense(env, MLP_LAYERS=[64,64], MLP_ACTIVATIONS=['relu', 'relu'], ACTOR_FINAL_ACTIVATION=None, NN_INIT='orthogonal', ACTOR_DIST_LOG_STD=0.0)
    model_output = ppo_algorithm(env, model)