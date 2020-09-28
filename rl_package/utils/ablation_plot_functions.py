import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from rl_package.utils.plot_functions import get_n_expmt
import os

def get_key_string(comb_array):
    str_keys = []
    for array in comb_array:
        str_key = ''
        for i in array:
            str_key = str_key + str(int(i))
        str_keys.append(str_key)
    return str_keys

def get_file_names(path, key_str):
    dirs = os.listdir(path)
    file_names = []
    for key in key_str:
        for folder in dirs:
            if key in folder:
                file_names.append(folder)
                break
    return file_names

def get_file_paths(path, comb_array):
    key_str = get_key_string(comb_array)
    file_names = get_file_names(path, key_str)
    file_paths = []
    for file in file_names:
        file_paths.append(path+'/'+file)
    return file_paths

def get_mean_reward_array(file_paths):
    mean_reward_array = []
    for file_path in file_paths:
        n_expmt = get_n_expmt(file_path)
        r = []
        for i in range(n_expmt):
            data = np.loadtxt(file_path+'/log'+str(i)+'.csv', delimiter=',', skiprows=1)
            r.append(data[-1,1])
        mean_reward_array.append(np.mean(r))
    return mean_reward_array

def ablation_plot(path, False_cases=['LR annealing', 'grad clip', 'relu', 'orthogonal'], True_cases=['no LR annealing', 'no grad clip', 'tanh', 'xavier']):
    comb_array = list( product([0,1], repeat=len(True_cases)) )
    file_paths = get_file_paths(path, comb_array)
    mean_reward_array = get_mean_reward_array(file_paths)


    # plot ablation histogram
    fig, axs =  plt.subplots(len(True_cases),1)
    mean_reward_array = np.array(mean_reward_array)
    for i in range(len(True_cases)):
        ind = [ bool(comb_array[j][i]) for j in range( len(comb_array) ) ] 
        true_data = mean_reward_array[ind]
        false_data = mean_reward_array[np.invert(ind)]
        axs[i].hist((true_data, false_data), stacked=True)
        axs[i].legend([True_cases[i], False_cases[i]])
    fig.savefig(path+'/ablation_histograms.png', bbox_inches=0, pad_inches=0)
    print('ablation_histograms.png is saved at '+path)
    
