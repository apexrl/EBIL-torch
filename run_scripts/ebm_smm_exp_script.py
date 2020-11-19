import yaml
import argparse
import joblib
import numpy as np
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
print(sys.path)

import torch

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.envs.wrappers import ScaledEnv
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.ebil.energy_models.simple_ebm_models import MLPEBM
from rlkit.torch.ebil.energy_models.simple_ae_models import MLPAE
from rlkit.torch.state_marginal_matching.ebm_learn_smm import EBMLearn


def experiment(variant):
    with open('expert_demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read())
    demos_path = listings[variant['expert_name']]['file_paths'][variant['expert_idx']]
    buffer_save_dict = joblib.load(demos_path)
    target_state_buffer = buffer_save_dict['data']
    target_state_buffer /= variant['rescale']
    state_indices = torch.LongTensor(variant['state_indices'])

    env_specs = variant['env_specs']
    env = get_env(env_specs)
    env.seed(env_specs['eval_env_seed'])
    training_env = get_env(env_specs)
    training_env.seed(env_specs['training_env_seed'])

    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1
    
    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # build the policy models
    net_size = variant['policy_net_size']
    num_hidden = variant['policy_num_hidden_layers']
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    
    input_dim = len(state_indices)

    # build the energy model
    if (variant['ebm_params']['mode']) == 'deen':
        ebm_model = MLPEBM(
            input_dim,
            num_layer_blocks=variant['ebm_num_blocks'],
            hid_dim=variant['ebm_hid_dim'],
            hid_act=variant['ebm_hid_act'],
            use_bn=variant['ebm_use_bn'],
            clamp_magnitude=variant['ebm_clamp_magnitude']
        )

        algorithm = EBMLearn(
            env=env,
            training_env=training_env,
            ebm=ebm_model,
            input_dim = input_dim,
            exploration_policy=policy,
            sigma=variant['sigma'],

            target_state_buffer=target_state_buffer,
            state_indices=state_indices,
            **variant['ebm_params']
        )
    
    # build the energy model
    elif (variant['ebm_params']['mode']) == 'ae':
        ebm_model = MLPAE(
            input_dim,
            num_layer_blocks=variant['ebm_num_blocks'],
            hid_dim=variant['ebm_hid_dim'],
            hid_act=variant['ebm_hid_act'],
            use_bn=variant['ebm_use_bn'],
        )

        algorithm = EBMLearn(
            env=env,
            training_env=training_env,
            ebm=ebm_model,
            input_dim = input_dim,
            exploration_policy=policy,
            sigma=None,
            rescale=variant['rescale'],

            target_state_buffer=target_state_buffer,
            state_indices=state_indices,
            **variant['ebm_params']
        )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

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
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    exp_prefix = exp_prefix + '--sigma-{}'.format(exp_specs['sigma'])

    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, snapshot_mode="all")

    experiment(exp_specs)
