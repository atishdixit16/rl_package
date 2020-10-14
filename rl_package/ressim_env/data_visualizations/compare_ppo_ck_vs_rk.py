import numpy as np
import matplotlib.pyplot as plt
import torch

import rl_package
from rl_package.ppo_vanilla.ppo import test_env, ActorCritic
from rl_package.utils.set_seed import set_seed
from rl_package.ressim_env.env_templates import generate_environment

# comparison between single and multiple permeability model rewards for training and testing data
path_ck = '/home/ad181/RemoteDir/ablation_study/PPO/single_phase/CK/1ph-C-CK-full-1t_1000_0.001/model2'
path_rk = '/home/ad181/RemoteDir/ablation_study/PPO/single_phase/RK/1ph-C-RK-full-1t_1000_0.001/model2'

# reproducibility
set_seed(1)

# parameters for model
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

env = generate_environment('1ph-C-RK-full-1t')
num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
MLP_LAYERS=[1536,1536]
ACTOR_FINAL_ACTIVATION='sigmoid'
ACTOR_DIST_LOG_STD=-1.9
MLP_ACTIVATIONS=['relu', 'relu']
NN_INIT = 'orthogonal'

model_ck = ActorCritic(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, NN_INIT, std=ACTOR_DIST_LOG_STD).to(device)
model_ck.load_state_dict(torch.load(path_ck, map_location=map_location))

model_rk = ActorCritic(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, NN_INIT, std=ACTOR_DIST_LOG_STD).to(device)
model_rk.load_state_dict(torch.load(path_rk, map_location=map_location))

def get_RK_rewards(env, model, data='train', base_case=False):
    k_train = np.load(rl_package.__file__+'/rl_package/ressim_env/k_'+data+'_batch.npy')
    reward_array = []
    for i in range(10):
        state = env.reset()
        env.set_k(k_train[i])
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            dist, _ = model(state.to(device))
            action = dist.sample().cpu().numpy()[0]
            if base_case:
                action = 0.5
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        reward_array.append(total_reward)
    return reward_array

model_ck_train = np.array( [ get_RK_rewards(env, model_ck, data='train') for _ in range(10)] ).mean(axis=0)
model_rk_train = np.array( [ get_RK_rewards(env, model_rk, data='train') for _ in range(10)] ).mean(axis=0)
model_base_train = get_RK_rewards(env, model_ck, data='train', base_case=True)

model_ck_test = np.array( [ get_RK_rewards(env, model_ck, data='test') for _ in range(10)] ).mean(axis=0)
model_rk_test = np.array( [ get_RK_rewards(env, model_rk, data='test') for _ in range(10)] ).mean(axis=0)
model_base_test = get_RK_rewards(env, model_ck, data='test', base_case=True)

fig, axs =  plt.subplots(1,2, figsize=(8,4))

axs[0].plot(model_ck_train,'*--')
axs[0].plot(model_rk_train,'o--')
axs[0].plot(model_base_train, '.--')
axs[0].set_title('Training Permeability Data')
axs[0].grid('on')
axs[0].set_xlabel('permeability index')
axs[0].set_ylabel('oil recovery factor (%)')
axs[0].legend(['single permeability model', 'multiple permeability model', 'all open results'])

axs[1].plot(model_ck_test,'*--')
axs[1].plot(model_rk_test,'o--')
axs[1].plot(model_base_test, '.--')
axs[1].set_title('Testing Permeability Data')
axs[1].grid('on')
axs[1].set_xlabel('permeability index')
axs[1].set_ylabel('oil recovery factor (%)')
axs[1].legend(['single permeability model', 'multiple permeability model', 'all open results'])

fig.savefig(path_ck[:-6]+'/compare_ppo_ck_vs_rk.pdf',bbox_inhes='tight' )
plt.close()
print('compare_ppo_ck_vs_rk.pdf is saved at '+ path_ck[:-6] )