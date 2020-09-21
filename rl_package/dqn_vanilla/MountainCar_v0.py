from rl_package.dqn_vanilla.dqn import dqn_algorithm
import gym

for i in range(5):
    print('trial: {}'.format(i))
    model = \
    dqn_algorithm(ENV= gym.make('MountainCar-v0'),
                  NUM_ENV=8,
                  SEED=i,
                  TOTAL_TIMESTEPS = 200000,
                  GAMMA = 1.0,
                  MEMORY_SIZE = 50000,
                  BATCH_SIZE = 128,
                  EXPLORATION_MAX = 1.0,
                  EXPLORATION_MIN = 0.9,
                  EXPLORATION_FRACTION = 0.9,
                  TRAINING_FREQUENCY = 200,
                  FILE_PATH = 'MountainCar_results/',
                  SAVE_MODEL = True,
                  MODEL_FILE_NAME = 'model'+str(i),
                  LOG_FILE_NAME = 'log'+str(i),
                  TIME_FILE_NAME = 'time'+str(i),
                  PRINT_FREQ = 1000,
                  N_EP_AVG = 100,
                  VERBOSE = 'True',
                  MLP_LAYERS = [8,8],
                  MLP_ACTIVATIONS = ['relu','relu'],
                  LEARNING_RATE = 1e-3,
                  EPOCHS = 1,
                  GRAD_CLIP = False,
                  DOUBLE_DQN = False,
                  USE_TARGET_NETWORK = True,
                  TARGET_UPDATE_FREQUENCY = 200,
                  LOAD_WEIGHTS = False,
                  LOAD_WEIGHTS_MODEL_PATH = 'CartPole-v0/results/model'+str(i)+'.h5')