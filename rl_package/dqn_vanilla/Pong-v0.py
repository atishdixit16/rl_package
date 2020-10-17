from rl_package.dqn_vanilla.dqn import dqn_algorithm
from rl_package.utils.standard_nn_architectures import QNetworkCNN
from rl_package.utils.env_wrappers import MaxAndSkipEnv, FireResetEnv, ProcessFrame84, ImageToPyTorch, BufferWrapper, ScaledFloatFrame
from rl_package.utils.plot_functions import reward_plot
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
import gym

def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env) 
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)

for i in range(5):
    env = make_env('Pong-v0')
    env.seed(i)
    model = QNetworkCNN(env)
    model_output = \
    dqn_algorithm(ENV= env, MODEL=model,
                  NUM_ENV=8,
                  SEED=i,
                  TOTAL_TIMESTEPS = 2500000,
                  GAMMA = 0.99,
                  MEMORY_SIZE = 10000,
                  BATCH_SIZE = 32,
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
                  N_TEST_ENV = 40, #100,
                  VERBOSE = 'True',
                  LEARNING_RATE = 1e-3,
                  EPOCHS = 1,
                  GRAD_CLIP = True,
                  DOUBLE_DQN = True,
                  USE_TARGET_NETWORK = True,
                  TARGET_UPDATE_FREQUENCY = 1000)
