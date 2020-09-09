import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from rl_package.ressim_env.ressim_env import ResSimEnv
from rl_package.ressim_env.spatial_expcov import batch_generate

def get_rewards(env, k):
    _, done = env.reset(), False
    r=0
    env.set_k(k)
    while not done:
        _, reward, done, _ = env.step(5)
        r += reward
    return r

def generate_stratified_k_sampling(k_l, k_sigma):

    '''
    Generate stratified samples of permeability from distribution of 
    cumulative rewards for 1000 spatially correleated permeability.

    It generates two numpy datasets each considering 10 permeability samples:
    k_train_batch.npy: consist of 10 stratified permeability samples that can be used while training RL algorithm
    k_test_batch.npy: consist of 10 stratified permeability samples that can be used while testing RL algorithm

    parameters:
    k_l     - length scale of exponential covariance
    k_sigma - amplitude of exponential covariance

    *In order to use these permeability fields in ResSimEnv class, make sure to use k_type='random'

    '''
    env = ResSimEnv( nx=32, ny=32, lx=1.0, ly=1.0, 
                     k=1,k_type='uniform', mobility='quadratic', phi=0.2, 
                     mu_w=1.0, mu_o=2.0, 
                     s_wir=0.2, s_oir=0.2, 
                     dt=1e-3, nstep=10, terminal_step=10,
                     state_spatial_param='full', state_temporal_param=1, action_space='discrete')
    k_batch = batch_generate(nx=env.nx, ny=env.ny, length=k_l, sigma=k_sigma, lx=env.lx, ly=env.ly, sample_size=1000)
    k_batch = np.exp(k_batch)
    r_array = []
    for i in trange(1000):
        r_array.append(get_rewards(env,k_batch[i]))

    plt.hist(r_array)
    plt.xlabel('total_reward')
    plt.show()

    # stratified sampling
    _, bin_edges = np.histogram(r_array)
    bin_edges[0] = 0
    bin_edges[-1] = 100
    indices = np.digitize(r_array, bin_edges, right=True)

    train_idx = []
    test_idx = []

    for i in np.unique(indices):
        idxs = np.where(indices==i)
        train_idx.append(idxs[0][0])
        test_idx.append(idxs[0][-1])

    k_train_batch = []
    k_test_batch = []
    for i in range(len(train_idx)):
        k_train_batch.append(k_batch[train_idx[i]])
        k_test_batch.append(k_batch[test_idx[i]])

    np.save('k_train_batch.npy', np.array(k_train_batch) )
    np.save('k_test_batch.npy', np.array(k_test_batch) )
