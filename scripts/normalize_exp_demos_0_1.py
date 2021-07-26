'''
The normalization I'm using here is different than the one for the meta version
'''
import numpy as np
import joblib
import yaml
import os
from os import path as osp
import copy

from rlkit.core.vistools import plot_histogram
from rlkit.launchers import config


EXPERT_LISTING_YAML_PATH = 'expert_demos_listing.yaml'
LOCAL_DEMO_DIR = './demos/'

with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
    listings = yaml.load(f.read())

for i, expert in enumerate(
    [
        # 'norm_hopper_4_demos_sub_20',
        # 'norm_hopper_16_demos_sub_20',
        # 'norm_hopper_32_demos_sub_20',
        # 'norm_walker_4_demos_sub_20',
        # 'norm_walker_16_demos_sub_20',
        # 'norm_walker_32_demos_sub_20',
        # 'norm_ant_4_demos_sub_20',
        # 'norm_ant_16_demos_sub_20',
        # 'norm_ant_32_demos_sub_20',
        # 'norm_halfcheetah_4_demos_sub_20',
        # 'norm_halfcheetah_16_demos_sub_20',
        # 'norm_halfcheetah_32_demos_sub_20',
        'norm_hopper_4_demos'
    ]
):
    data_path = osp.join(listings[expert]['file_paths'][0])
    save_dir = osp.join(LOCAL_DEMO_DIR, 'norm01_'+expert)
    save_path = save_dir + '.pkl'

    buffer_save_dict = joblib.load(data_path)
    expert_replay_buffer = buffer_save_dict['train']

    data = buffer_save_dict['train']._observations
    std = buffer_save_dict['obs_std']
    mean = buffer_save_dict['obs_mean']

    minmax_train = copy.deepcopy(buffer_save_dict['train'])
    minmax_test = copy.deepcopy(buffer_save_dict['test'])

    raw_data = minmax_train._observations * std + mean
    max_data = np.max(raw_data, axis=0)
    min_data = np.min(raw_data, axis=0)
    buffer_save_dict['obs_max'] =  max_data
    buffer_save_dict['obs_min'] =  min_data
    norm_data = (raw_data - min_data) / (max_data - min_data)
    np.nan_to_num(norm_data, copy=False)

    minmax_train._observations = norm_data
    buffer_save_dict['norm_train'] = minmax_train

    print(save_path)
    joblib.dump(buffer_save_dict, save_path, compress=3)

  

print('\n\nRemember to add the new normalized demos to your expert listings!\n\n')
