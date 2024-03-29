import random
import numpy as np
import os
import time
from collections import deque
import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from rl_package.utils.set_seed import set_seed
from rl_package.utils.multiprocessing_env import SubprocVecEnv
from rl_package.utils.env_wrappers import ParallelEnvWrapper


class DQNSolver:

    def __init__(self,
                 env,
                 model,
                 target_model,
                 optimizer,
                 scheduler,
                 device,
                 EPOCHS,
                 USE_TARGET_NETWORK,
                 LOSS_TYPE,
                 GRAD_CLIP,
                 LR_ANNEAL,
                 DOUBLE_DQN,
                 TOTAL_TIMESTEPS,
                 MEMORY_SIZE,
                 BATCH_SIZE,
                 GAMMA,
                 EXPLORATION_MAX,
                 EXPLORATION_MIN,
                 EXPLORATION_FRACTION):
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.exploration_rate = EXPLORATION_MAX
        self.exploration_max = EXPLORATION_MAX
        self.exploration_min = EXPLORATION_MIN
        self.exploration_fraction = EXPLORATION_FRACTION
        self.epochs = EPOCHS
        self.use_target_network = USE_TARGET_NETWORK
        self.grad_clip = GRAD_CLIP
        self.lr_anneal = LR_ANNEAL
        self.loss_type = LOSS_TYPE
        self.double_dqn =  DOUBLE_DQN
        self.total_timesteps = TOTAL_TIMESTEPS
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.action_space = env.action_space.n
        self.loss = 1.0

        self.create_memoery_buffer(memory_size=MEMORY_SIZE)

    def create_memoery_buffer(self, memory_size):
        self.states = deque(maxlen=memory_size)
        self.actions = deque(maxlen=memory_size)
        self.rewards = deque(maxlen=memory_size)
        self.next_states = deque(maxlen=memory_size)
        self.dones = deque(maxlen=memory_size)

    def append_memory_buffer(self, state, action, reward, next_state, done):
        for st,act,rew,n_st,dn in zip(state, action, reward, next_state, done):
            self.states.append(st)
            self.actions.append(act)
            self.rewards.append(rew)
            self.next_states.append(n_st)
            self.dones.append(dn)

    def sample_batch(self, batch_size):
        memory_len = len(self.states)
        indices = random.sample(range(memory_len), batch_size)
        state_list = [ self.states[i] for i in indices ]
        action_list = [ self.actions[i] for i in indices ]
        reward_list = [ self.rewards[i] for i in indices ]
        next_state_list = [self.next_states[i] for i in indices]
        done_list = [self.dones[i] for i in indices]

        return state_list, action_list, reward_list, next_state_list, done_list


    def remember(self, state, action, reward, next_state, done):
        self.append_memory_buffer( state, action, reward, next_state, done)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        q_values = q_values.cpu().detach().numpy()
        actions = np.argmax(q_values, axis=1) # deterministic actions
        for i in range(actions.shape[0]):
            if random.random() < self.exploration_rate:
                actions[i] = random.randrange(self.action_space) #stochastic actions
        return actions

    def experience_replay(self):
        if len(self.states) < self.batch_size:
            return
        ### new code for network training
        state, action, reward, state_next, done = self.sample_batch(batch_size=self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        state_next = torch.FloatTensor(state_next).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_t = self.model(state)
        if self.use_target_network:
            q_t1 = self.target_model(state_next)
        else:
            q_t1 = self.model(state_next)
        q_t1_best = torch.max(q_t1, dim=1)[0]
        if self.double_dqn and self.use_target_network:
            q_t1_local = self.model(state_next)
            ind = np.argmax(q_t1_local.cpu().detach().numpy(), axis=1)
        for i in range(self.batch_size):
            if self.double_dqn and self.use_target_network:
                q_t1_best[i] = q_t1[i,ind[i]]
            q_t[i,int(action[i])] = reward[i] + self.gamma*(1-done[i])*q_t1_best[i]
        # train the DQN network
        for _ in range(self.epochs):
            q_output = self.model(state)
            if self.loss_type=='mse':
                criterion = nn.MSELoss()
            if self.loss_type=='huber':
                criterion = nn.SmoothL1Loss()
            loss = criterion(q_output, q_t.detach())
            self.optimizer.zero_grad()
            if self.grad_clip:
                nn.utils.clip_grad_value_(self.model.parameters(), 0.5)
            loss.backward()
            self.optimizer.step()
            if self.lr_anneal:
                self.scheduler.step()
        self.loss=loss.cpu().detach().numpy()

    def eps_timestep_decay(self, t):
        fraction = min (float(t)/int(self.total_timesteps*self.exploration_fraction), 1.0)
        self.exploration_rate = self.exploration_max + fraction * (self.exploration_min - self.exploration_max)

    def update_target_network(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)

def test_env(env, model, device, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state.to(device))
        next_state, reward, done, _ = env.step( np.argmax(q_values.cpu().detach().numpy()[0]) )
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
            state = torch.FloatTensor(state).to(device)
            q_values = model(state)
            actions = np.argmax(q_values.cpu().detach().numpy() , axis=1) 
            state, reward, done, _ = envs.step(list(actions))
            total_reward += reward
        mean_return.append(np.mean(total_reward))
    return np.mean(mean_return)


def dqn_algorithm(ENV, MODEL, NUM_ENV=8,
                  TOTAL_TIMESTEPS = 100000, GAMMA = 0.95, MEMORY_SIZE = 1000, BATCH_SIZE = 32,
                  EXPLORATION_MAX = 1.0, EXPLORATION_MIN = 0.02, EXPLORATION_FRACTION = 0.7,
                  TRAINING_FREQUENCY = 1000, DOUBLE_DQN = False, USE_TARGET_NETWORK = True, TARGET_UPDATE_FREQUENCY = 5000,
                  N_TEST_ENV = 200, TEST_ENV_FUNC = test_env_mean_return,
                  LEARNING_RATE = 1e-3,  EPOCHS = 1, LOSS_TYPE = 'huber',
                  GRAD_CLIP = False, LR_ANNEAL = False,
                  VERBOSE = 'False', FILE_PATH = 'results/', SAVE_MODEL = False, MODEL_CHECKPOINT=10000, PRINT_FREQ = 100,
                  MODEL_FILE_NAME = 'model', LOG_FILE_NAME = 'log', TIME_FILE_NAME = 'time',
                  SEED=1):

    '''
    DQN Algorithm parameters

    env : environment class object
    num_env : no. for environment vectorization (multiprocessing env)
    total_timesteps : Total number of timesteps
    training_frequency : frequency of training (experience replay)
    gamma : discount factor : 
    buffer_size : Replay buffer size 
    batch_size : batch size for experience replay 
    exploration_max : maximum exploration at the begining 
    exploration_min : minimum exploration at the end 
    exploration_fraction : fraction of total timesteps on which the exploration decay takes place 
    output_folder : output filepath 
    save_model : boolean to specify whether the model is to be saved 
    model_file_name : name of file to save the model at the end learning 
    log_file_name : name of file to store DQN results 
    time_file_name : name of file to store computation time 
    print_frequency : results printing episodic frequency 
    n_ep_avg : no. of episodes to be considered while computing average reward 
    verbose : print episodic results 
    learning_rate : learning rate for the neural network 
    epochs : no. of epochs in every experience replay 
    grad_clip : boolean to specify whether to use gradient clipping in the optimizer (graclip value 0.5) 
    lr_anneal : boolean to specify whether to use learning rate annealing (linear wrt timestep) 
    double_dqn : boolean to specify whether to employ double DQN 
    use_target_network : boolean to use target neural network in DQN 
    target_update_frequency : timesteps frequency to do weight update from online network to target network 
    load_weights : boolean to specify whether to use a prespecified model to initializa the weights of neural network 
    load_weights_model_path : path for the model to use for weight initialization 
    '''

    before = time.time()
    num_envs = NUM_ENV

    assert not TOTAL_TIMESTEPS % NUM_ENV, 'Invalid total timesteps. For convinience, select such that TOTAL_TIMESTEPS % NUM_ENV = 0'
    assert not PRINT_FREQ % NUM_ENV, 'Invalid print frequency. For convinience, select such that PRINT_FREQ % NUM_ENV = 0'
    assert not MODEL_CHECKPOINT % NUM_ENV, 'Invalid model checkpoint frequency. For convinience, select such that MODEL_CHECKPOINT % NUM_ENV = 0'
    assert not TRAINING_FREQUENCY % NUM_ENV, 'Invalid training frequency. For convinience, select such that TRAINING_FREQUENCY % NUM_ENV = 0'
    assert not N_TEST_ENV % NUM_ENV, 'Invalid no. of test env samples. For convinience, select such that N_TEST_ENV % NUM_ENV = 0'
    assert LOSS_TYPE in ['mse', 'huber'], 'Invalidt LOSS_TYPE. Should be one of these: mse, huber'

    if TOTAL_TIMESTEPS%NUM_ENV:
        print('Error: total timesteps is not divisible by no. of envs')
        return 

    def make_env():
        def _thunk():
            ENV.seed(SEED)
            return ENV

        return _thunk

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    env = ENV
    env.seed(SEED)

    # for reproducibility
    set_seed(SEED)

    t = 0
    explore_percent, mean100_rew, steps, NN_tr_loss = [],[],[],[]

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    model = MODEL.to(device)
    target_model = copy.deepcopy(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lam = lambda steps: 1-t/TOTAL_TIMESTEPS
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)

    dqn_solver = DQNSolver(env,
                           model,
                           target_model,
                           optimizer,
                           scheduler,
                           device,
                           EPOCHS,
                           USE_TARGET_NETWORK,
                           LOSS_TYPE,
                           GRAD_CLIP,
                           LR_ANNEAL,
                           DOUBLE_DQN,
                           TOTAL_TIMESTEPS,
                           MEMORY_SIZE,
                           BATCH_SIZE,
                           GAMMA,
                           EXPLORATION_MAX,
                           EXPLORATION_MIN,
                           EXPLORATION_FRACTION)

    while True:
        state = envs.reset()
        while True:
            t += num_envs
            dqn_solver.eps_timestep_decay(t)
            action = dqn_solver.act(state)
            state_next, reward, terminal, _ = envs.step(action)
            dqn_solver.remember(state, action, reward, state_next, terminal)
            if t%TRAINING_FREQUENCY==0:
                dqn_solver.experience_replay()
            state = state_next
            if (t%PRINT_FREQ==0):
                test_reward = TEST_ENV_FUNC(envs, model, device, n_trials=N_TEST_ENV)
                explore_percent.append(dqn_solver.exploration_rate*100)
                mean100_rew.append(test_reward)
                steps.append(t)
                NN_tr_loss.append(dqn_solver.loss)
                if VERBOSE:
                    print('Exploration %: '+str(int(explore_percent[-1]))+', Mean_reward: '+str(round( mean100_rew[-1], 4) )+', timestep: '+str(t)+', tr_loss: '+str(np.round(NN_tr_loss[-1],4)) )

            if t>TOTAL_TIMESTEPS:
                output_table = np.stack((steps, mean100_rew, explore_percent, NN_tr_loss))
                if not os.path.exists(FILE_PATH):
                    os.makedirs(FILE_PATH)
                file_name = str(FILE_PATH)+LOG_FILE_NAME+'.csv'
                np.savetxt(file_name, np.transpose(output_table), delimiter=',', header='Timestep,Rewards,Exploration %,Training Score')
                after = time.time()
                time_taken = after-before
                np.save( str(FILE_PATH)+TIME_FILE_NAME, time_taken )
                if SAVE_MODEL:
                    torch.save(dqn_solver.model.state_dict(), FILE_PATH+MODEL_FILE_NAME )
                return dqn_solver.model
            if SAVE_MODEL and t%MODEL_CHECKPOINT==0:
                if not os.path.exists(FILE_PATH):
                    os.makedirs(FILE_PATH)
                torch.save(dqn_solver.model.state_dict(), FILE_PATH+MODEL_FILE_NAME )
            if USE_TARGET_NETWORK and t%TARGET_UPDATE_FREQUENCY==0:
                dqn_solver.update_target_network()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    import gym
    from rl_package.utils.standard_nn_architectures import QNetworkDense


    parser = argparse.ArgumentParser()

    # DQN algorithms parameters
    parser.add_argument('--num_env', type=int, default=8, help='no. for environment vectorization')
    parser.add_argument('--seed', type=int, default=1, help='seed for pseudo random generator')
    parser.add_argument('--total_timesteps', type=int, default=200000, help='Total number of timesteps')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_size',  type=int, default=1000, help='Replay buffer size')
    parser.add_argument('--batch_size',  type=int, default=128, help='batch size for experience replay')
    parser.add_argument('--exploration_max',  type=float, default=1.0, help='maximum exploration at the begining')
    parser.add_argument('--exploration_min',  type=float, default=0.02, help='minimum exploration at the end')
    parser.add_argument('--exploration_fraction',  type=float, default=0.6, help='fraction of total timesteps on which the exploration decay takes place')
    parser.add_argument('--output_folder', default='results_temp/', help='output filepath')
    parser.add_argument('--save_model', type=str2bool, default=False,  help='boolean to specify whether the model is to be saved')
    parser.add_argument('--model_file_name', default='model', help='name of file to save the model at the end learning')
    parser.add_argument('--log_file_name', default='log', help='name of file to store DQN results')
    parser.add_argument('--time_file_name', default='time', help='name of file to store computation time')
    parser.add_argument('--print_frequency',  type=int, default=1000, help='printing with timestep frequency')
    parser.add_argument('--n_test_env',  type=int, default=200, help='no. of episodes to be considered while computing average reward')
    parser.add_argument('--verbose', type=str2bool, default=True,  help='print episodic results')
    parser.add_argument('--mlp_layers', nargs='+', type=int, default=[64, 64], help='list of neurons in each hodden layer of the DQN network')
    parser.add_argument('--mlp_activations', nargs='+', default=['relu', 'relu'], help='list of activation functions in each hodden layer of the DQN network')
    parser.add_argument('--nn_init', default='orthogonal', help='neural network initialization: othogonal or xavier')
    parser.add_argument('--learning_rate',  type=float, default=1e-3, help='learning rate for the neural network')
    parser.add_argument('--epochs',  type=int, default=1, help='no. of epochs in every experience replay')
    parser.add_argument('--grad_clip', type=str2bool, default=False,  help='boolean to specify whether to use gradient clipping in the optimizer (graclip value 10.0)')
    parser.add_argument('--lr_anneal', type=str2bool, default=False,  help='boolean to specify whether to use gradient learning rate annealing (linear wrt timesteps)')
    parser.add_argument('--double_dqn', type=str2bool, default=False,  help='boolean to specify whether to employ double DQN')
    parser.add_argument('--use_target_network', type=str2bool, default=True,  help='boolean to use target neural network in DQN')
    parser.add_argument('--target_update_frequency',  type=int, default=1000, help='timesteps frequency to do weight update from online network to target network')
    parser.add_argument('--training_frequency',  type=int, default=200, help='timesteps frequency to train the DQN (experience replay)')
    parser.add_argument('--load_weights', type=str2bool, default=False,  help='boolean to specify whether to use a prespecified model to initializa the weights of neural network')
    parser.add_argument('--load_weights_model_path', default='results/model0.h5', help='path for the model to use for weight initialization')
    args = parser.parse_args()
    
    '''
    # List of parameters

    # DQN algorithm parameters
    ENV_NAME = args.env_name
    NUM_ENV = args.num_env
    GAMMA = args.gamma
    TOTAL_TIMESTEPS = args.total_timesteps
    MEMORY_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size
    TARGET_UPDATE_FREQUENCY = args.target_update_frequency

    # saving/loggin parameters
    PRINT_FREQ = args.print_frequency
    N_EP_AVG = args.n_ep_avg
    SAVE_MODEL = args.save_model
    FILE_PATH = args.output_folder
    MODEL_FILE_NAME = args.model_file_name
    LOG_FILE_NAME = args.log_file_name
    TIME_FILE_NAME = args.time_file_name
    VERBOSE = args.verbose

    # DQNSolver parameters
    EPOCHS = args.epochs
    GRAD_CLIP = args.grad_clip
    MLP_LAYERS = args.mlp_layers
    MLP_ACTIVATIONS = args.mlp_activations
    DOUBLE_DQN = args.double_dqn
    LEARNING_RATE = args.learning_rate
    EXPLORATION_MAX = args.exploration_max
    EXPLORATION_MIN = args.exploration_min
    EXPLORATION_FRACTION = args.exploration_fraction
    USE_TARGET_NETWORK = args.use_target_network
    LOAD_WEIGHTS = args.load_weights
    LOAD_WEIGHTS_MODEL_PATH = args.load_weights_model_path
    '''

    env = gym.make('CartPole-v0')
    model = QNetworkDense(env, MLP_LAYERS=[64,64], MLP_ACTIVATIONS=['relu','relu'], NN_INIT='orthogonal')
    model_output = \
    dqn_algorithm(ENV=env, MODEL=model, NUM_ENV=args.num_env,
                  SEED=args.seed,
                  GAMMA = args.gamma,
                  TOTAL_TIMESTEPS = args.total_timesteps, MEMORY_SIZE = args.buffer_size, BATCH_SIZE = args.batch_size,
                  TRAINING_FREQUENCY = args.training_frequency, TARGET_UPDATE_FREQUENCY = args.target_update_frequency,
                  N_TEST_ENV = args.n_test_env,
                  VERBOSE = args.verbose,
                  PRINT_FREQ = args.print_frequency,
                  SAVE_MODEL = args.save_model,
                  FILE_PATH = args.output_folder,
                  MODEL_FILE_NAME = args.model_file_name,
                  LOG_FILE_NAME = args.log_file_name,
                  TIME_FILE_NAME = args.time_file_name,
                  EPOCHS = args.epochs,
                  GRAD_CLIP = args.grad_clip,
                  LR_ANNEAL= args.lr_anneal,
                  DOUBLE_DQN = args.double_dqn,
                  LEARNING_RATE = args.learning_rate,
                  EXPLORATION_MAX = args.exploration_max,
                  EXPLORATION_MIN = args.exploration_min,
                  EXPLORATION_FRACTION = args.exploration_fraction,
                  USE_TARGET_NETWORK = args.use_target_network)
