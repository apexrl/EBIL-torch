import yaml
import argparse
import joblib
import numpy as np
import os,sys,inspect
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.ebil.energy_models.simple_ebm_models import MLPEBM
from rlkit.torch.ebil.ebil import EBIL
from rlkit.envs.wrappers import ScaledEnv
from rlkit.launchers import config
import torch

ebm_id_dic = {'sigma':{'0.05':'0000', '0.1':'0001', '0.15':'0002', '0.2':'0003', '0.3':'0004'}}

def experiment(variant):
    with open('expert_demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read())
    expert_demos_path = listings[variant['expert_name']]['file_paths'][variant['expert_idx']]
    buffer_save_dict = joblib.load(expert_demos_path)
    expert_replay_buffer = buffer_save_dict['train']

    env_specs = variant['env_specs']
    env = get_env(env_specs)
    env.seed(env_specs['eval_env_seed'])
    training_env = get_env(env_specs)
    training_env.seed(env_specs['training_env_seed'])

    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))
    
    if variant['scale_env_with_demo_stats']:
        env = ScaledEnv(
            env,
            obs_mean=buffer_save_dict['obs_mean'],
            obs_std=buffer_save_dict['obs_std'],
            acts_mean=buffer_save_dict['acts_mean'],
            acts_std=buffer_save_dict['acts_std'],
        )
        training_env = ScaledEnv(
            training_env,
            obs_mean=buffer_save_dict['obs_mean'],
            obs_std=buffer_save_dict['obs_std'],
            acts_mean=buffer_save_dict['acts_mean'],
            acts_std=buffer_save_dict['acts_std'],
        )

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
    qf1 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    
    #TODO: Load state-only EBM
    # build the energy model
    if variant['ebil_params']['mode'] == 'deen':
        """
        ebm_model = MLPEBM(
            obs_dim + action_dim if not variant['ebil_params']['state_only'] else 2*obs_dim,
            num_layer_blocks=variant['ebm_num_blocks'],
            hid_dim=variant['ebm_hid_dim'],
            hid_act=variant['ebm_hid_act'],
            use_bn=variant['ebm_use_bn'],
            clamp_magnitude=variant['ebm_clamp_magnitude'],
        )
        """
        ebm_exp_name = 'ebm-deen-'+variant['env_specs']['env_name']+'-'+str(variant['expert_traj_num'])+'-train'
        ebm_dir = os.path.join(config.LOCAL_LOG_DIR, ebm_exp_name)

        ebm_id_dirs = os.listdir(ebm_dir)
        ebm_id = ebm_id_dic['sigma'][str(variant['ebm_sigma'])]
        ebm_id_dirs = [_ for _ in ebm_id_dirs if ebm_id in _]
        ebm_id_dirs = sorted(ebm_id_dirs, key=lambda x: os.path.getmtime(os.path.join(ebm_dir, x)))

        load_ebm_dir = os.path.join(ebm_dir, ebm_id_dirs[-1]) # Choose the last as the load ebm dir
        load_epoch = variant['ebm_epoch']
        load_name = 'itr_{}.pkl'.format(load_epoch)
        if load_epoch == 'best':
            load_name = 'best.pkl'
        load_ebm_path =  os.path.join(load_ebm_dir, load_name)
        
        load_ebm_pkl = joblib.load(load_ebm_path, mmap_mode='r')
        ebm_model = load_ebm_pkl['ebm']

        print("loaded EBM from {}".format(load_ebm_path))

        
        # Test
        if variant['test']:
            batch_data = expert_replay_buffer.random_batch(100, keys=['observations', 'actions'])
            obs = torch.Tensor(batch_data['observations'])
            acts = torch.Tensor(batch_data['actions'])
            exp_input = torch.cat([obs, acts], dim=1)
            print("Not expert data", ebm_model(exp_input*200).mean().item())
            print("Expert data", ebm_model(exp_input).mean().item())
            exit(1)
    
    elif variant['ebil_params']['mode'] == 'ae':
        ebm_exp_name = 'ebm-ae-'+variant['env_specs']['env_name']+'-'+str(variant['expert_traj_num'])+'-train'
        ebm_dir = os.path.join(config.LOCAL_LOG_DIR, ebm_exp_name)

        ebm_id_dirs = os.listdir(ebm_dir)
        ebm_id_dirs = sorted(ebm_id_dirs, key=lambda x: os.path.getmtime(os.path.join(ebm_dir, x)))

        load_ebm_dir = os.path.join(ebm_dir, ebm_id_dirs[-1]) # Choose the last as the load ebm dir
        load_epoch = variant['ebm_epoch']
        load_name = 'itr_{}.pkl'.format(load_epoch)
        if load_epoch == 'best':
            load_name = 'best.pkl'
        load_ebm_path =  os.path.join(load_ebm_dir, load_name)
        
        load_ebm_pkl = joblib.load(load_ebm_path, mmap_mode='r')
        ebm_model = load_ebm_pkl['ebm']

        print("loaded EBM from {}".format(load_ebm_path))

        
        # Test
        if variant['test']:
            batch_data = expert_replay_buffer.random_batch(100, keys=['observations', 'actions'])
            obs = torch.Tensor(batch_data['observations'])
            acts = torch.Tensor(batch_data['actions'])
            exp_input = torch.cat([obs, acts], dim=1).to(ptu.device)
            print("Not expert data", ebm_model(exp_input*200).mean().item())
            print("Expert data", ebm_model(exp_input).mean().item())
            exit(1)

    else:
        raise NotImplementedError
    
    # set up the algorithm
    trainer = SoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['sac_params']
    )
    algorithm = EBIL(
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        rew_func = variant['rew_func'],
        cons = variant['cons'],

        ebm=ebm_model,
        policy_trainer=trainer,
        expert_replay_buffer=expert_replay_buffer,
        **variant['ebil_params']
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
