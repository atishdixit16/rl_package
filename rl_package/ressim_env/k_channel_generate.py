import numpy as np

'''
generates permeability field with a linear channel across the grid
start and end of the channel are selected with uniform distribution

parameters:
nx, ny, lx, ly: grid dimensions
channel_k: permeability value at the channel
base_k: permeability value outside the channel
channel_width: width of the permeability channel
sample_size: number of realizations generated
seed: seed for reproducibility

'''

def get_channel_end_indices(nx=32, ny=32, lx=1.0, ly=1.0, channel_width=0.125, seed=1):
    assert channel_width<ly, 'invalid channel width. condition violated: channel_width < ly'
    channel_left_end = np.random.uniform(0,ly-channel_width)
    channel_right_end = np.random.uniform(0,ly-channel_width)
    return channel_left_end, channel_right_end


def single_generate(nx=32,ny=32,lx=1.0,ly=1.0,channel_k=1.0, base_k=0.01, channel_width=0.125, channel_left_end=0.4375, channel_right_end=0.4375):
    index_left = round(channel_left_end*ny)
    index_right = round(channel_right_end*ny)
    grid_channel_width = round(channel_width*ny)
    k = base_k*np.ones((nx,ny))
    for i in range(nx):
        j = ( (index_right - index_left) / nx ) *i + index_left
        for w in range(grid_channel_width ):
            k[round(j)+w, i] = channel_k
    return k


def batch_generate(nx=32, ny=32, lx=1.0, ly=1.0, channel_k=1.0, base_k=0.01, channel_width=0.125, sample_size=10, seed=1):
    np.random.seed(seed) #for reproducibility
    k_batch = []
    for _ in range(sample_size):
        channel_left_end, channel_right_end = get_channel_end_indices(nx, ny, lx, ly, channel_width, seed)
        k = single_generate(nx,ny,lx,ly,channel_k, base_k, channel_width, channel_left_end, channel_right_end)
        k_batch.append(k)
    return np.array(k_batch)

def generate_train_data():
    top=0.125
    mid=0.4375
    bottom=0.75
    k0 = single_generate(nx=32,ny=32,lx=1.0,ly=1.0,channel_k=1.0, base_k=0.01, channel_width=0.125, channel_left_end=mid, channel_right_end=mid)
    k1 = single_generate(nx=32,ny=32,lx=1.0,ly=1.0,channel_k=1.0, base_k=0.01, channel_width=0.125, channel_left_end=top, channel_right_end=top)
    k2 = single_generate(nx=32,ny=32,lx=1.0,ly=1.0,channel_k=1.0, base_k=0.01, channel_width=0.125, channel_left_end=bottom, channel_right_end=bottom)
    k3 = single_generate(nx=32,ny=32,lx=1.0,ly=1.0,channel_k=1.0, base_k=0.01, channel_width=0.125, channel_left_end=top, channel_right_end=bottom)
    k4 = single_generate(nx=32,ny=32,lx=1.0,ly=1.0,channel_k=1.0, base_k=0.01, channel_width=0.125, channel_left_end=bottom, channel_right_end=top)
    return np.array([k0, k1, k2, k3, k4])
