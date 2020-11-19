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
from rlkit.envs.wrappers import ScaledEnv
from rlkit.launchers import config
import torch
from rlkit.torch.state_marginal_matching.ebil_smm import EBIL

ebm_id_dic_pm_infty = {'sigma':{'0.03':'0000', '0.04':'0001', '0.05':'0002'}}
ebm_id_dic_pm_x = {'sigma':{'0.05':'0000', '0.08':'0001', '0.1':'0002'}}
ebm_id_dic_pm_triangle = {'sigma':{'0.05':'0000', '0.08':'0001', '0.1':'0002'}}
ebm_id_dic_pm_square = {'sigma':{'0.05':'0000', '0.08':'0001', '0.1':'0002'}}
ebm_id_dics = {'simple_point_mass_infty':ebm_id_dic_pm_infty, 'simple_point_mass_x':ebm_id_dic_pm_x, 'simple_point_mass_square':ebm_id_dic_pm_square, 'simple_point_mass_triangle':ebm_id_dic_pm_triangle}


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
        ebm_exp_name = 'ebm-deen-smm-implementation-'+variant['env_specs']['task_name']
        ebm_dir = os.path.join(config.LOCAL_LOG_DIR, ebm_exp_name)

        ebm_id_dirs = os.listdir(ebm_dir)
        tmp = []
        ebm_id_dic = ebm_id_dics[variant['env_specs']['env_name']+'_'+variant['env_specs']['task_name']]

        if str(variant['ebm_sigma']) in ebm_id_dic['sigma'].keys():
            ebm_id = ebm_id_dic['sigma'][str(variant['ebm_sigma'])]
            tmp = [_ for _ in ebm_id_dirs if ebm_id in _]
        else:
            raise NotImplementedError

        if len(tmp)>0:
            ebm_id_dirs = tmp
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
        
    else:
        raise NotImplementedError

    # Test
    if variant['test']:
        batch_data = target_state_buffer / variant['rescale']
        obs = torch.Tensor(batch_data[:1000]).to(ptu.device) 
        print("Not expert data", ebm_model(obs*200).mean().item())
        print("Expert data", ebm_model(obs).mean().item())
        exit(1)

    
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
        rescale = variant['rescale'],

        ebm=ebm_model,
        policy_trainer=trainer,
        target_state_buffer=target_state_buffer,
        state_indices=state_indices,
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
    exp_prefix = exp_prefix + '--ebm_sigma-{}--ebm_epoch-{}--rs-{}'.format(exp_specs['ebm_epoch'],
                                                            exp_specs['ebm_sigma'],
                                                            exp_specs['sac_params']['reward_scale'],)
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed, snapshot_mode="all")

    experiment(exp_specs)
