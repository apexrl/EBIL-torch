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

import matplotlib.pyplot as plt

def experiment(variant):
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

        
    else:
        raise NotImplementedError

    # Test
    if variant['test']:
        batch_data = target_state_buffer
        obs = torch.Tensor(batch_data[:100])
        exp_input = torch.cat([obs, acts], dim=1).to(ptu.device)
        print("Not expert data", ebm_model(exp_input*200).mean().item())
        print("Expert data", ebm_model(exp_input).mean().item())
        exit(1)

    x = np.linspace(-1.25,20, 1000)
    y = np.linspace(-1.25,10, 1000)
    
    rewards = []
    for i in range(1000):
        data = []
        for j in range(1000):
            coords = np.array((x[j], y[i]))   
            data.append((x[j], y[i])) 

        data = np.array(data) / variant['rescale']
        data = torch.Tensor(data).to(ptu.device)
        reward = ebm_model(data).squeeze().detach().cpu().numpy() 
        rewards.append(reward)
    #data = np.array(data)
    #data = torch.Tensor(data).to(ptu.device)

    rewards = np.array(rewards)
    print(rewards.shape)
    # rewards = np.reshape(rewards, (1000,1000))

    fig,ax=plt.subplots(figsize=(6,6))
    # im = ax.imshow(rewards, cmap=plt.cm.hot_r)
    # plt.colorbar(im)
    h=plt.contourf(rewards, cmap=plt.cm.hot_r)
    cb=plt.colorbar(h)
    ax.set_xticks([58, 293, 528, 764, 999])                                                        
    ax.set_xticklabels(['0','5', '10', '15', '20'])
    ax.set_yticks([111, 555, 999])                                                        
    ax.set_yticklabels(['0','5', '10'])
    plt.savefig('./figs/'+variant['env_specs']['env_name']+'_'+variant['env_specs']['task_name']+'_'+str(variant['ebm_sigma'])+'_'+str(variant['ebm_epoch'])+'.pdf', bbox_inches='tight')

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
    exp_prefix = exp_prefix + '--ebm_sigma-{}--ebm_epoch-{}'.format(exp_specs['ebm_epoch'],
                                                            exp_specs['ebm_sigma'])
    seed = exp_specs['seed']
    set_seed(seed)

    experiment(exp_specs)
