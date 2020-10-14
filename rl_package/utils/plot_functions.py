import numpy as np
import matplotlib.pyplot as plt
import os


def get_n_expmt(path): #get no. of log#.csv files in the directory 
    dirs = np.array(os.listdir( str (path) ))
    for i in range(dirs.size):
        dirs[i] = dirs[i][:3]
    n_expmt = len (np.where(dirs=='log')[0] )
    assert n_expmt > 0, 'no log#.csv files found in the directory: '+path
    return n_expmt

def get_xy_data(paths):
    data = []
    for path in paths:
        n_expmt = get_n_expmt(path)
        case_data = []
        for i in range(n_expmt):
            case_data.append( np.loadtxt(path+'/log'+str(i)+'.csv', delimiter=',', skiprows=1) )
        data.append(case_data)

    n = len(paths)
    xs, ys = [],[]
    for i in range(n):
        xs.append(data[i][0][:,0])
        n_cases = len(data[i])
        y = []
        for j in range(n_cases):
            y.append(data[i][j][:,1])
        ys.append(y)
    return xs,ys


def reward_plot(paths, case_titles, plot_type='median'):
    assert plot_type in ['median', 'mean'], 'Invalid plot type. Should be one of these: median, mean.'

    xs, ys = get_xy_data(paths)

    if plot_type=='median':
        for x,y in zip(xs,ys):
            plt.plot(x, np.nanmedian(y, axis=0) )
            plt.fill_between(x, np.nanpercentile(y, 25, axis=0), np.nanpercentile(y, 75, axis=0), alpha=0.25)
        plt.grid('True')
        plt.legend(case_titles)
        plt.xlabel('Timesteps')
        plt.ylabel('Rewards')
        plt.savefig(paths[0]+'/rewards_median.png')
        plt.close()
        print('reward_median.png is saved at {}'.format(paths[0]))
    
    if plot_type=='mean':
        for x,y in zip(xs,ys):
            plt.plot(x, np.nanmean(y, axis=0) )
        plt.grid('True')
        plt.legend(case_titles)
        plt.xlabel('Timesteps')
        plt.ylabel('Rewards')
        plt.savefig(paths[0]+'/rewards_mean.png')
        plt.close()
        print('reward_mean.png is saved at {}'.format(paths[0]))

