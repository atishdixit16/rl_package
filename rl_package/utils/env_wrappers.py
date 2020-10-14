import numpy as np
import gym
import collections
import cv2


# source: openAI standard wrappers 
# from article: https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
       '''
       some ATARI games as Pong require a user to press the FIRE button to start the game. 
       The following code corresponds to the wrapper FireResetEnvthat presses the FIRE 
       button in environments that require that for the game to start
       '''
       super(FireResetEnv, self).__init__(env)
       assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
       assert len(env.unwrapped.get_action_meanings()) >= 3
       
    def step(self, action):
       return self.env.step(action)
    
    def reset(self):
       self.env.reset()
       obs, _, done, _ = self.env.step(1)
       if done:
          self.env.reset()
       obs, _, done, _ = self.env.step(2)
       if done:
          self.env.reset()
       return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        '''
        On one hand, it allows us to speed up significantly the training by applying max to N 
        observations (four by default) and returns this as an observation for the step. This is 
        because on intermediate frames, the chosen action is simply repeated and we can make an 
        action decision every N steps as processing every frame with a Neural Network is quite 
        a demanding operation, but the difference between consequent frames is usually minor.

        On the other hand, it takes the maximum of every pixel in the last two frames and using 
        it as an observation. Some Atari games have a flickering effect (when the game draws different 
        portions of the screen on even and odd frames, a normal practice among Atari 2600 developers to 
        increase the complexity of the game’s sprites), which is due to the platform’s limitation. 
        For the human eye, such quick changes are not visible, but they can confuse a Neural Network.
        '''
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
           obs, reward, done, info = self.env.step(action)
           self._obs_buffer.append(obs)
           total_reward += reward
           if done:
               break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
       self._obs_buffer.clear()
       obs = self.env.reset()
       self._obs_buffer.append(obs)
       return obs

def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        '''
        every frame is scaled down from 210x160, with three color frames (RGB color channels), 
        to a single-color 84 x84 image using a colorimetric grayscale conversion. Different 
        approaches are possible. One of them is cropping non-relevant parts of the image and 
        then scaling down as is done in the following code:
        '''
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def reset(self):
        return _process_frame84(self.env.reset())


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        '''
        the class BufferWrapper stacks several (usually four) subsequent frames together
        '''
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space =  gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),old_space.high.repeat(n_steps, axis=0), dtype=dtype)
    
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())
    
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        '''
        This simple wrapper changes the shape of the observation from HWC (height, width, channel) 
        to the CHW (channel, height, width) format required by PyTorch
        '''
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,            
                                shape=(old_shape[-1], 
                                old_shape[0], old_shape[1]),
                                dtype=np.float32)
    
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    '''
    The screen obtained from the emulator is encoded as a tensor of bytes with values from 0 to 255, 
    which is not the best representation for an NN. So, we need to convert the image into floats and 
    rescale the values to the range [0.0…1.0]. This is done by the ScaledFloatFrame wrapper
    '''
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


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

# example 
if __name__ == "__main__":
    def make_env(env_name):
        env = gym.make(env_name)
        env = MaxAndSkipEnv(env)
        env = FireResetEnv(env)
        env = ProcessFrame84(env)
        env = ImageToPyTorch(env) 
        env = BufferWrapper(env, 4)
        return ScaledFloatFrame(env)
    
    env= make_env('Pong-v0')

    done, state = False, env.reset()
    i = 0
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        i = i+1
        print('step: {}, state: {}, reward: {}, done: {}'.format(i, state.shape, reward, done))