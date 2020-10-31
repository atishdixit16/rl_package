import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import functools
from collections import deque

import rl_package
from rl_package.ressim_env.ressim import Grid, SaturationEquation, PressureEquation
from rl_package.ressim_env.utils import linear_mobility, quadratic_mobility, lamb_fn, f_fn, df_fn

class PermSensEnv():

    def __init__(self,
                 grid, k, phi, # domain properties
                 dt, nstep, terminal_step, # timesteps
                 q, s, # initial conditions
                 k_list, seed): #env paramters

        # for reproducibility
        self.seed(seed)

        # domain properties
        self.grid=grid
        self.k = k
        self.phi = phi
        if k_list is not None:
            self.k = np.random.choice(k_list.shape[0])

        # timesteps
        self.dt = dt  # timestep
        self.nstep = nstep # no. of timesteps solved in one episodic step
        self.terminal_step = terminal_step # terminal step in episode
        self.episode_step = 0

        # initial conditions
        self.Q = np.sum(q[q>0])
        self.q_init = q # storing inital values for reset function
        self.q = q
        self.s = s

        # env parameters
        self.k_list = k_list

        # original oil in place
        self.ooip = self.grid.lx * self.grid.ly * self.phi[0,0] * (1)

        # RL parameters
        # state
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        high = np.array([1e5]*self.state.shape[0])
        self.observation_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        
        # action
        self.action_space = spaces.Discrete(3)

    def f_fn(self,s): return s
    
    def df_fn(self,s): return np.ones(len(s))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_q_mapping(self, action):
        assert action in (0,1,2), 'Invalid action. Should be one of these: 0,1,2'
        if action == 0:
            self.q[0,-1] = -self.Q/2
            self.q[-1,0] = -self.Q/2
        if action == 1:
            self.q[0,-1] = 0.0
            self.q[-1,0] = -self.Q
        if action == 2:
            self.q[0,-1] = -self.Q
            self.q[-1,0] = 0.0

        return self.q

    def step(self, action):

        self.q = self.action_to_q_mapping(action)

        # solve pressure
        self.solverP = PressureEquation(self.grid, q=self.q, k=self.k)
        self.solverS = SaturationEquation(self.grid, q=self.q, phi=self.phi, s=self.s_load, f_fn=self.f_fn, df_fn=self.df_fn)

        oil_pr = 0.0

        for _ in range(self.nstep):
            # solve pressure
            self.solverP.s = self.s_load
            self.solverP.step()
            # solve saturation
            self.solverS.v = self.solverP.v
            self.solverS.step(self.dt)
            self.s_load = self.solverS.s
            # oil_pr = oil_pr + (-self.q[0,0] * (1 - self.f_fn( self.s_load[0,0]) ) + -self.q[-1,0] * ( 1 - self.f_fn( self.s_load[-1,0]) ) )*self.dt # oil production
            oil_pr = oil_pr + -np.sum( self.q[self.q<0] * ( 1- self.f_fn(self.s_load[self.q<0]) ) )*self.dt

        # state
        self.state = self.s_load.reshape(-1)

        #reward
        reward = oil_pr / self.ooip # recovery rate
        reward = reward*100 # in percentage

        # done
        self.episode_step += 1
        if self.episode_step >= self.terminal_step:
            done=True
        else:
            done=False

        return self.state, reward, done, {}

    def set_k(self, k):
        self.k = k

    def reset(self):

        self.q = self.q_init

        if self.k_list is not None:
            self.k = np.random.choice(self.k_list.shape[0])

        self.episode_step = 0

        self.s_load = self.s
        self.state = self.s_load.reshape(-1)

        return self.state

    def render(self):
        pass

    def close(self):
        pass