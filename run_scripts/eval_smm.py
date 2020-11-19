import yaml
import argparse
import joblib
import numpy as np
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core import eval_util

from rlkit.envs.wrappers import ScaledEnv, MinmaxEnv
from rlkit.samplers import PathSampler
from rlkit.torch.sac.policies import MakeDeterministic

import matplotlib.pyplot as plt

import torch

def experiment(variant):
    with open('expert_demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read())
    demos_path = listings[variant['expert_name']]['file_paths'][variant['expert_idx']]
    print(demos_path)
    buffer_save_dict = joblib.load(demos_path)
    target_state_buffer = buffer_save_dict['data']
    # target_state_buffer /= variant['rescale']
    state_indices = torch.LongTensor(variant['state_indices'])

    env_specs = variant['env_specs']
    env = get_env(env_specs)
    env.seed(env_specs['eval_env_seed'])

    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))

    policy = joblib.load(variant['policy_checkpoint'])['exploration_policy']
    if variant['eval_deterministic']:
        policy = MakeDeterministic(policy)
    policy.to(ptu.device)

    eval_sampler = PathSampler(
        env,
        policy,
        variant['num_eval_steps'],
        variant['max_path_length'],
        no_terminal=variant['no_terminal'],
        render=variant['render'],
        render_kwargs=variant['render_kwargs']
    )
    test_paths = eval_sampler.obtain_samples()
    obs = []
    for path in test_paths:
        obs += path['observations']
    x = [o[0] for o in obs]
    y = [o[1] for o in obs]
    
    fig,ax=plt.subplots(figsize=(6,6))
    plt.scatter(x,y)
    plt.xlim(-1.25, 20)
    plt.ylim(-1.25, 10)
    ax.set_yticks([0, 5, 10])
    ax.set_xticks([0, 5, 10, 15, 20])
    plt.savefig('./figs/'+variant['env_specs']['task_name']+'.pdf', bbox_inches='tight')

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    parser.add_argument('-g', '--gpu', help='gpu id', type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    # make all seeds the same.
    exp_specs['env_specs']['eval_env_seed'] = exp_specs['env_specs']['training_env_seed'] = exp_specs['seed']

    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
