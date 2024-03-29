import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from rl_package.utils.plot_functions import get_n_expmt, get_xy_data
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

def get_reward_array(file_paths):
    reward_array = []
    for file_path in file_paths:
        n_expmt = get_n_expmt(file_path)
        r = []
        for i in range(n_expmt):
            data = np.loadtxt(file_path+'/log'+str(i)+'.csv', delimiter=',', skiprows=1)
            r.append(data[-1,1])
        reward_array.append(r)
    return reward_array

def get_mean_reward_array(file_paths):
    reward_array = get_reward_array(file_paths)
    return np.mean(np.array(reward_array), axis=1 )

def ablation_plot(path, False_cases=['LR annealing', 'grad clip', 'relu', 'orthogonal'], True_cases=['no LR annealing', 'no grad clip', 'tanh', 'xavier']):
    comb_array = list( product([0,1], repeat=len(True_cases)) )
    file_paths = get_file_paths(path, comb_array)
    reward_array = get_reward_array(file_paths)


    # plot ablation histogram
    fig, axs =  plt.subplots(len(True_cases),2, figsize=(10,16))
    reward_array = np.array(reward_array)
    for i in range(len(True_cases)):
        ind = [ bool(comb_array[j][i]) for j in range( len(comb_array) ) ] 
        true_data = reward_array[ind].reshape(-1)
        false_data = reward_array[np.invert(ind)].reshape(-1)
        axs[i,0].hist((false_data, true_data), stacked=True, density=True,cumulative=-1  )
        axs[i,0].legend([ False_cases[i], True_cases[i] ] )
        axs[i,0].grid('on')
        axs[i,0].set_xlabel('reward')
        axs[i,0].set_ylabel('1 - CDF(reward)')

    # plot effect of each parameter
    comb_array = np.array(comb_array)
    sum_array = np.sum(comb_array, axis=1)
    file_paths_filtered = np.array(file_paths)[sum_array<=1]
    xs_base, ys_base = get_xy_data([file_paths_filtered[0]])
    effect_file_paths = np.flip(file_paths_filtered[1:])
    xs, ys = get_xy_data(effect_file_paths)

    for i in range(len(True_cases)):
        axs[i,1].plot(xs_base[0], np.nanmedian(ys_base[0], axis=0) )
        axs[i,1].fill_between(xs_base[0], np.nanpercentile(ys_base[0], 25, axis=0), np.nanpercentile(ys_base[0], 75, axis=0), alpha=0.25)
        axs[i,1].plot(xs[i], np.nanmedian(ys[i], axis=0) )
        axs[i,1].fill_between(xs[i], np.nanpercentile(ys[i], 25, axis=0), np.nanpercentile(ys[i], 75, axis=0), alpha=0.25)
        axs[i,1].legend([False_cases[i], True_cases[i]])
        axs[i,1].grid('on')
        axs[i,1].set_xlabel('timesteps')
        axs[i,1].set_ylabel('mean reward')
    fig.savefig(path+'/ablation_report_plots.pdf',bbox_inhes='tight' )
    plt.close()
    print('ablation_report_plots.pdf is saved at '+path)

    # plot best test case
    mean_reward_array = get_mean_reward_array(file_paths)
    best_results_file_path = file_paths[np.argmax(mean_reward_array)]
    best_combination = comb_array[np.argmax(mean_reward_array)]
    xs,ys = get_xy_data([best_results_file_path])

    fig, axs = plt.subplots(1,1, figsize=(10,10))
    axs.plot(xs[0], np.nanmedian(ys[0], axis=0) )
    axs.fill_between(xs[0], np.nanpercentile(ys[0], 25, axis=0), np.nanpercentile(ys[0], 75, axis=0), alpha=0.25)
    axs.grid('on')
    axs.set_xlabel('timesteps')
    axs.set_ylabel('mean reward')
    cases = np.vstack( (False_cases, True_cases) ) 
    axs.set_title(best_results_file_path+'\n'+ cases[best_combination[0],0] + '-'  + cases[best_combination[1],1] + '-' + cases[best_combination[2],2] + '-' + cases[best_combination[3],3] )

    fig.savefig(path+'/ablation_best_results_plots.pdf',bbox_inhes='tight' )
    plt.close()
    print('ablation_best_results_plots.pdf is saved at '+path)