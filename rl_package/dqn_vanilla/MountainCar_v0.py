from rl_package.dqn_vanilla.dqn import dqn_algorithm
from rl_package.utils.standard_nn_architectures import QNetworkDense
import gym

for i in range(5):
    print('trial: {}'.format(i))
    env = gym.make('MountainCar-v0')
    model =  QNetworkDense(env, MLP_LAYERS=[64,64], MLP_ACTIVATIONS=['tanh','tanh'], NN_INIT='orthogonal', seed=i)
    model_output = \
    dqn_algorithm(ENV= env, MODEL=model,
                  NUM_ENV=8,
                  SEED=i,
                  TOTAL_TIMESTEPS = 200000,
                  GAMMA = 1.0,
                  MEMORY_SIZE = 50000,
                  BATCH_SIZE = 10000,
                  EXPLORATION_MAX = 1.0,
                  EXPLORATION_MIN = 0.4,
                  EXPLORATION_FRACTION = 0.9,
                  TRAINING_FREQUENCY = 200,
                  FILE_PATH = 'MountainCar_results/',
                  SAVE_MODEL = True,
                  MODEL_FILE_NAME = 'model'+str(i),
                  LOG_FILE_NAME = 'log'+str(i),
                  TIME_FILE_NAME = 'time'+str(i),
                  PRINT_FREQ = 1000,
                  N_TEST_ENV=96,
                  VERBOSE = 'True',
                  LEARNING_RATE = 1e-4,
                  EPOCHS = 1,
                  GRAD_CLIP = False,
                  DOUBLE_DQN = False,
                  USE_TARGET_NETWORK = True,
                  TARGET_UPDATE_FREQUENCY = 200)