from rl_package.dqn_vanilla.dqn import dqn_algorithm
from rl_package.utils.standard_nn_architectures import QNetworkDense
from rl_package.utils.plot_functions import reward_plot
import gym

for i in range(5):
    env = gym.make('CartPole-v0')
    env.seed(i)
    model = QNetworkDense(env, MLP_LAYERS=[64,64], MLP_ACTIVATIONS=['relu','relu'], NN_INIT='orthogonal', seed=i)
    model_output = \
    dqn_algorithm(ENV= env, MODEL=model,
                  NUM_ENV=8,
                  SEED=i,
                  TOTAL_TIMESTEPS = 250000,
                  GAMMA = 0.95,
                  MEMORY_SIZE = 1000,
                  BATCH_SIZE = 128,
                  EXPLORATION_MAX = 1.0,
                  EXPLORATION_MIN = 0.02,
                  EXPLORATION_FRACTION = 0.6,
                  TRAINING_FREQUENCY = 200,
                  FILE_PATH = 'rl_package/dqn_vanilla/CartPole_results/',
                  SAVE_MODEL = True,
                  MODEL_FILE_NAME = 'model'+str(i),
                  LOG_FILE_NAME = 'log'+str(i),
                  TIME_FILE_NAME = 'time'+str(i),
                  PRINT_FREQ = 5000,
                  N_TEST_ENV = 96, #100,
                  VERBOSE = 'True',
                  LEARNING_RATE = 1e-3,
                  EPOCHS = 1,
                  GRAD_CLIP = False,
                  DOUBLE_DQN = False,
                  USE_TARGET_NETWORK = True,
                  TARGET_UPDATE_FREQUENCY = 1000)

reward_plot( [ '/home/ad181/rl_package/rl_package/dqn_vanilla/CartPole_results'], ['DQN Cartpole-v0']  )