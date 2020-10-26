import numpy as np

from rl_package.ressim_env.ressim_env import ResSimEnv
from rl_package.ressim_env.spatial_expcov import batch_generate

'''

template naming convention:
phase # : 1ph for single phase, 
          2ph for two phase
action space : C for coninous (for policy based algorithms), 
               D for discrete (for value based algorithms)
permeability : CK for contant permeability, 
               RK for random permeability
state spatial param : full for all saturation values, 
                      well for saturation values at well location, 
                      vic for saturation values in the vicinity of wells
state temporal param : 1t for a single step saturation values
                       2t for last two steps saturation values
                       3t for last three steps saturation values

For example, the template 2ph-D-CK-well-3t refers to an environment with 
                - two phase flow
                - discrete action space
                - contant permeability value
                - state represented with saturations at well locations
                - state represented with last three steps of the episode

'''

def generate_environment(template, 
                         nx=32, ny=32, lx=1.0, ly=1.0, 
                         k=1, phi=0.2, k_l=0.1, k_sigma=1.0, 
                         s_wir=0.2, s_oir = 0.2, mu_w = 1.0, mu_o = 2.0,
                         seed=1):

    np.random.seed(seed)

    template = template.split('-')
    assert template[0] in ['1ph','2ph'] , 'invalid phase #, should be one of these: 1ph, 2ph'
    assert template[1] in ['D','C'] , 'invalid action space param, should be one of these: C, D'
    assert template[2] in ['CK','RK'] , 'invalid permeability param, should be one of these: CK, RK'
    assert template[3] in ['full','vic','well'] , 'invalid spatial state param, should be one of these: full, well, vic'
    assert template[4] in ['1t','2t', '3t'] , 'invalid temporal state param, should be one of these: 1t, 2t, 3t'

    if template[0]=='1ph':
        s_wir = s_oir = 0.0
        mu_w = mu_o = 1.0
        mobility = 'linear'
        dt, nstep, terminal_step = 1e-2, 1, 10 
    else:
        mobility = 'quadratic'
        dt, nstep, terminal_step = 1e-3, 10, 10 

    if template[1]=='D':
        action_space = 'discrete'
    else:
        action_space = 'continous'

    if template[2]=='CK':
        k_type = 'constant'
    else:
        k_type = 'random'

    state_spatial_param = template[3]
    state_temporal_param = int(template[4][0])
    
    env = ResSimEnv( nx, ny, lx, ly, 
                 k, k_type, mobility, phi, 
                 mu_w, mu_o, 
                 s_wir, s_oir, 
                 dt, nstep, terminal_step,
                 state_spatial_param, state_temporal_param, action_space, seed)

    return env

def test_env(env, action):
    state, done = env.reset(), False
    i=r=0
    while not done:
        state, reward, done, _ = env.step(action)
        i += 1
        r += reward
        print('step: {}, state_shape: {}, action: {}, cum_reward: {}, done: {}'.format(i, state.shape, action, int(r), done))


if __name__ == "__main__":
    # example 1
    print('\nExample 1 : 2ph-D-CK-well-3t \n')
    env = generate_environment('2ph-D-CK-well-3t')
    test_env(env,5)

    # example 2
    print('\nExample 2 : 1ph-D-CK-full-1t \n')
    env = generate_environment('1ph-D-CK-full-1t')
    test_env(env,5)

    # example 3
    print('\nExample 3 : 2ph-C-CK-vic-2t \n')
    env = generate_environment('2ph-C-CK-vic-2t')
    test_env(env,0.5)
