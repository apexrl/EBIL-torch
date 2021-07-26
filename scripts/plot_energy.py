import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

traj_nums = 4
env = 'hopper'

# suffix = '2020_08_29_14_09_19_0004--s-0'

dir_name = '../logs/' + 'ebm-deen-{}-{}-train/'.format(env, traj_nums)#  + 'ebm_{}_{}_train_'.format(env, traj_nums)

if env in ['infty', 'spiral']:
    dir_name = '../logs/' + 'ebm-deen-smm-implementation-{}/'.format(env)

files = os.listdir(dir_name)
files = [_ for _ in files if 'ebm' in _]

for suffix in files:
    file_name = dir_name + suffix + '/progress.csv'
    df = pd.read_csv(file_name)

    random_energy = df['random_avg_energy']
    expert_energy = df['expert_avg_energy']
    exprandom_energy = df['expert*20_avg_energy']

    epoch = df['Epoch']

    plt.plot(epoch, random_energy, color='r')
    plt.plot(epoch, exprandom_energy, color='g')
    plt.plot(epoch, expert_energy, color='b')

    if not os.path.exists('../figs'):
        os.makedirs('../figs')
    plt.savefig('../figs/' + suffix + '.pdf')
    plt.cla()
