import gym
from gym import spaces

import numpy as np
import functools
from collections import deque

from rl_package.ressim_env.ressim import Grid, SaturationEquation, PressureEquation
from rl_package.ressim_env.utils import linear_mobility, quadratic_mobility, lamb_fn, f_fn, df_fn

class ResSimEnv():

    def __init__(self,
                 nx, ny, lx, ly,
                 k, k_type, mobility, phi,
                 mu_w, mu_o,
                 s_wir, s_oir,
                 dt, nstep, terminal_step,
                 state_spatial_param, state_temporal_param, action_space,
                 seed=1):

        # for reproducibility
        np.random.seed(seed)

        # grid
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.grid = Grid(nx=self.nx, ny=self.ny, lx=self.lx, ly=self.ly)

        # properties
        self.k_value = k; self.k_type = k_type
        self.k = self.choose_k(self.k_type, self.k_value) 
        self.phi = np.ones(self.grid.shape)*phi  # uniform porosity
        self.mu_w, self.mu_o = mu_w, mu_o  # viscosities
        self.s_wir, self.s_oir = s_wir, s_oir  # irreducible saturations

        # timestep
        self.dt = dt  # timestep
        self.nstep = nstep # no. of timesteps solved in one episodic step
        self.terminal_step = terminal_step # terminal step in episode
        self.episode_step = 0

        # original oil in place
        self.ooip = self.grid.lx * self.grid.ly * self.phi[0,0] * (1 - self.s_wir-self.s_oir)

        # source/sink
        self.q = np.zeros(self.grid.shape)
        self.q[0,0]=-0.5 # producer 1
        self.q[-1,0]=-0.5 # producer 2
        self.q[0,-1]=1.0 # injector 1
        self.s = np.ones(self.grid.shape) * self.s_wir  # initial water saturation
        self.s_load = self.s

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
        self.mask = self.get_state_mask(state_spatial_param) # spatial mask on full snapshot s_load
        self.state_que = self.get_state_que(state_temporal_param) # queue to consider states from previous episode steps

        self.state = self.state_processing()
        high = np.array([1e5]*self.state.shape[0])
        self.observation_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        
        # action
        self.action_space_param = action_space
        if self.action_space_param=='discrete':
            self.action_space = spaces.Discrete(11) 
        elif self.action_space_param=='continous':
            self.action_space = spaces.Box(low=np.array([0], dtype=np.float64), high=np.array([1], dtype=np.float64), dtype=np.float64)
        else:
            raise Exception('invalid action space. should be one of these: discrete or continous')

    def step(self, action):
        
        if self.action_space_param=='discrete':
            # source term for producer 1: q[0,0]
            self.q[0,0] = ( -1 / ( self.action_space.n - 1 ) ) * action
            self.q[-1,0] = -1 - self.q[0,0] # since q[0,0] + q[-1,0] = -1
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self.q[0,0] = - action
            self.q[-1,0] = -1 - self.q[0,0] # since q[0,0] + q[-1,0] = -1


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
            self.solverS.step_implicit(self.dt)
            self.s_load = self.solverS.s
            oil_pr = oil_pr + (-self.q[0,0] * (1 - self.f_fn( self.s_load[0,0]) ) + -self.q[-1,0] * ( 1 - self.f_fn( self.s_load[-1,0]) ) )*self.dt # oil production

        # state
        self.state = self.state_processing()

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

    def get_state_mask(self, state_spatial_param):
        # get the state mask when we initiate the object
        if state_spatial_param=='well':
            mask = np.zeros(self.grid.shape); mask[0,0]=mask[-1,0]=mask[0,-1]=1
        elif state_spatial_param=='vic':
            mask = np.zeros(self.grid.shape); mask[:3,:3]=mask[-3:,:3]=mask[:3,-3:]=1
        elif state_spatial_param=='full':
            mask = np.ones(self.grid.shape)
        else:
            raise Exception('invalid state_spatial_param. should be one of these: well, vic, full')
        return mask

    def state_spatial_processing(self):
        # masking the values of full snapshot 's_load' according to spatial_state_param
        stensil = np.ma.masked_equal(self.mask,1)
        self.state = self.s_load[stensil.mask]
        return self.state

    def get_state_que(self, state_temporal_param):
        # create a queue of the initial state which can be appended later on
        assert state_temporal_param in [1,2,3], 'invalid state_temporal_param. should ne one of these: 1,2,3'
        state_que = deque(maxlen=state_temporal_param)
        for _ in range(state_temporal_param):
            state_que.append(self.state_spatial_processing())
        return state_que
    
    def state_processing(self):
        # append the spatially processed state in the queue
        self.state_que.append(self.state_spatial_processing())
        return np.array(self.state_que).reshape(-1)

    def get_sw(self, x_ind, y_ind):
        return self.s_load[x_ind, y_ind]

    def choose_k(self, k_type, k):
        if k_type=='uniform':
            self.k = k*np.ones(self.grid.shape)
        elif k_type=='random':
            k_train = np.load('k_train_batch.npy')
            self.k = k_train[ np.random.randint(k_train.shape[0]) ] 
        else:
            raise Exception('invalid k_type. should be one of these: uniform or random')
        return self.k

    def set_k(self, k):
        self.k = k

    def reset(self):

        self.q[0,0]=-0.5 # producer 1
        self.q[-1,0]=-0.5 # producer 2
        self.q[0,-1]=1.0 # injector 1

        self.k = self.choose_k(self.k_type, self.k_value) 

        self.episode_step = 0

        self.s_load = self.s
        self.state = self.state_processing()

        return self.state

    def render(self):
        pass

    def close(self):
        pass
