import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import functools
from collections import deque

import rl_package
from rl_package.ressim_env.ressim import Grid, SaturationEquation, PressureEquation
from rl_package.ressim_env.utils import linear_mobility, quadratic_mobility, lamb_fn, f_fn, df_fn

class ResSimEnv_v1():

    def __init__(self,
                 grid, k, phi, s_wir, s_oir, # domain properties
                 mu_w, mu_o, mobility, # fluid properties
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
        self.s_wir = s_wir
        self.s_oir = s_oir

        # fluid properties
        self.mu_w = mu_w
        self.mu_o = mu_o
        assert mobility in ['linear', 'quadratic'], 'invalid mobility parameter. should be one of these: linear, quadratic'
        self.mobility = mobility

        # timesteps
        self.dt = dt  # timestep
        self.nstep = nstep # no. of timesteps solved in one episodic step
        self.terminal_step = terminal_step # terminal step in episode
        self.episode_step = 0

        # initial conditions
        self.q = q
        self.s = s

        # env parameters
        self.k_list = k_list

        # original oil in place
        self.ooip = self.grid.lx * self.grid.ly * self.phi[0,0] * (1 - self.s_wir-self.s_oir)

        # Model function (mobility and fractional flow function)
        if mobility=='linear':
            self.mobi_fn = functools.partial(linear_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        elif mobility=='quadratic':
            self.mobi_fn = functools.partial(quadratic_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        else:
            raise Exception('invalid mobility input. should be one of these: linear or quadratic')
        self.lamb_fn = functools.partial(lamb_fn, mobi_fn=self.mobi_fn)  # total mobility function
        self.f_fn = functools.partial(f_fn, mobi_fn=self.mobi_fn)  # water fractional flow function
        self.df_fn = functools.partial(df_fn, mobi_fn=self.mobi_fn)

        # RL parameters
        # state
        self.s_load = self.s
        self.state = self.s_load.reshape(-1)
        high = np.array([1e5]*self.state.shape[0])
        self.observation_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        
        # action
        self.Q = np.sum(self.q[q>0])                                # total flow across the field
        self.n_inj = self.q[self.q>0].shape[0]                      # no of injectors
        self.i_x, self.i_y =  np.where(q>0)[0], np.where(q>0)[1]    # injector co-ordinates
        self.n_prod = self.q[self.q<0].shape[0]                     # no of producers
        self.p_x, self.p_y =  np.where(q<0)[0], np.where(q<0)[1]    # producer co-ordinates
        self.action_space = spaces.Box(low=np.array([0]*(self.n_inj+self.n_prod), dtype=np.float64), 
                                       high=np.array([1]*(self.n_inj+self.n_prod), dtype=np.float64), 
                                       dtype=np.float64)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_q_mapping(self, action):
        action = np.clip(action,0,1)
        inj_flow = action[:self.n_inj] / np.sum(action[:self.n_inj])
        inj_flow = self.Q * inj_flow
        prod_flow = action[self.n_inj:] / np.sum(action[self.n_inj:])
        prod_flow = -self.Q * prod_flow

        q = np.zeros(self.grid.shape)
        i=0
        for x,y in zip(self.i_x, self.i_y):
            q[x,y] = inj_flow[i]
            i=i+1
        i=0
        for x,y in zip(self.p_x, self.p_y):
            q[x,y] = prod_flow[i]
            i=i+1
        return q


    def step(self, action):

        self.q = self.action_to_q_mapping(action)        

        # solve pressure
        self.solverP = PressureEquation(self.grid, q=self.q, k=self.k, lamb_fn=self.lamb_fn)
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

        self.q[0,0]=-0.5 # producer 1
        self.q[-1,0]=-0.5 # producer 2
        self.q[0,-1]=1.0 # injector 1

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
