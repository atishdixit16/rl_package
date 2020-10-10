import numpy as np
from rl_package.utils.multiprocessing_env import SubprocVecEnv
# import gym

# num_envs = 8
# env_name = "CartPole-v0"

# def make_env():
#     def _thunk():
#         env = gym.make(env_name)
#         env.seed(1)
#         return env

#     return _thunk

# envs = [make_env() for i in range(num_envs)]
# envs = SubprocVecEnv(envs)

class ParallelEnvWrapper():
    '''
    synchronised parallel environment operations
    step function returns zero reward when the episode terminates in a corresponding env vector
    '''
    def __init__(self,envs):
        self.envs = envs
        self.record_done = np.array([False]*self.envs.nenvs)

    def reset(self):
        self.record_done = np.array([False]*self.envs.nenvs)
        return self.envs.reset()

    def step(self,actions):
        s,r,d,i = self.envs.step(actions)
        r[self.record_done] = 0.0
        self.record_done[d] = True
        if self.record_done.any():
            return s,r,self.record_done,i
        return s,r,d,i
