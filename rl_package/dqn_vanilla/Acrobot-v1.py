from rl_package.dqn_vanilla.dqn_pytorch import dqn_algorithm
from rl_package.utils.plot_functions import reward_plot
import gym

for i in range(5):
    print('trial {}'.format(i))
    model = \
    dqn_algorithm(ENV= gym.make('Acrobot-v1'),
                  NUM_ENV=6,
                  SEED=i,
                  TOTAL_TIMESTEPS = 1000000,
                  GAMMA = 0.95,
                  MEMORY_SIZE = 10000,
                  BATCH_SIZE = 128,
                  EXPLORATION_MAX = 1.0,
                  EXPLORATION_MIN = 0.02,
                  EXPLORATION_FRACTION = 0.6,
                  TRAINING_FREQUENCY = 200,
                  FILE_PATH = 'rl_package/dqn_vanilla/Acrobot_results/',
                  SAVE_MODEL = True,
                  MODEL_FILE_NAME = 'model'+str(i),
                  LOG_FILE_NAME = 'log'+str(i),
                  TIME_FILE_NAME = 'time'+str(i),
                  PRINT_FREQ = 5000,
                  N_EP_AVG = 100,
                  VERBOSE = 'True',
                  MLP_LAYERS = [16,16],
                  MLP_ACTIVATIONS = ['relu','relu'],
                  LEARNING_RATE = 1e-3,
                  EPOCHS = 1,
                  GRAD_CLIP = False,
                  LR_ANNEAL= True,
                  NN_INIT='orthogonal',
                  DOUBLE_DQN = False,
                  USE_TARGET_NETWORK = True,
                  TARGET_UPDATE_FREQUENCY = 1000)

reward_plot( [ 'rl_package/dqn_vanilla/Acrobot_results'], ['DQN Acrobot-v1']  )