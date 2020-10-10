from rl_package.ppo_vanilla.ppo import ppo_algorithm
import gym

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')

    for i in range(5):
        model = ppo_algorithm(env, NUM_ENV=8,
                  TOTAL_STEPS=400000, NSTEPS=64, MINIBATCH_SIZE=128, N_EPOCH=30,
                  CLIP_PARAM=0.1, VF_COEF=0.5, ENT_COEF=0.001,
                  GAMMA=0.99, LAMBDA=0.95,
                  MLP_LAYERS=[64,64], MLP_ACTIVATIONS=['relu', 'relu'], LEARNING_RATE=1e-3,
                  PRINT_FREQ=8000, N_TEST_ENV=96, 
                  SAVE_RESULTS=True, FILE_PATH='pendulum_results/', LOG_FILE_NAME='log'+str(i), SAVE_MODEL=True, MODEL_FILE_NAME='model'+str(i),
                  SEED=i)