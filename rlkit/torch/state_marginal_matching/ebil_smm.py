import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm

def reward_func(rew_name, cons=0):
    if rew_name == '-energy':
        return lambda x:-x+cons
    elif rew_name == '-energy+1':
        return lambda x:1-x
    elif rew_name == '-energy-1':
        return lambda x:-1-x
    elif rew_name == 'exp(-energy-1)':
        return lambda x:torch.exp(-x-1)
    elif rew_name == '(-energy+1)div2':
        return lambda x:(-x+1)/2.0
    elif rew_name == 'exp(-energy+1)':
        return lambda x:torch.exp(-x+1)
    elif rew_name == 'exp(-energy)':
        return lambda x:torch.exp(-x)
    else:
        raise NotImplementedError


class EBIL(TorchBaseAlgorithm):
    '''
        Energy-Based Imitaion Learning.
    '''
    def __init__(
        self,

        mode, # deen
        rew_func,
        ebm,
        policy_trainer,

        target_state_buffer,
        state_indices, # LongTensor state indices for matching marginals

        state_only=False,
        cons=0, 
        clamp_magnitude=1.0,
        rescale=1.0,

        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,

        num_policy_update_loops_per_train_call=1,
        num_policy_updates_per_loop_iter=100,

        use_grad_pen=True,
        grad_pen_weight=10,

        **kwargs
    ):
        assert mode in ['deen', 'ae'], 'Invalid ebil algorithm!'
        if kwargs['wrap_absorbing']: raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.rew_func = rew_func
        self.cons = cons
        self.state_only = state_only
        self.clamp_magnitude = clamp_magnitude

        self.target_state_buffer = target_state_buffer
        self.state_indices = state_indices

        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert
        
        self.ebm = ebm
        self.rescale = rescale
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.num_policy_update_loops_per_train_call = num_policy_update_loops_per_train_call
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.ebm_eval_statistics = None

    def get_batch(self, batch_size, from_target_state_buffer, keys=None):
        if from_target_state_buffer:
            buffer = self.target_state_buffer
            batch = {
                'observations': buffer[np.random.choice(buffer.shape[0], size=batch_size)]
            }
        else:
            buffer = self.replay_buffer
            batch = buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def get_energy(self, data):
        if self.mode == 'deen':
            return self.ebm(data)
        elif self.mode == 'ae':
            return (data - self.ebm(data))**2
        else:
            raise NotImplementedError

    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        super()._end_epoch()


    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.ebm_eval_statistics)
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        super().evaluate(epoch)


    def _do_training(self, epoch):
        for t in range(self.num_policy_update_loops_per_train_call):
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)

    def _do_policy_training(self, epoch):
        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert, False)
            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert, True)
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k]
                    ],
                    dim=0
                )
        else:
            policy_batch = self.get_batch(self.policy_optim_batch_size, False)
        
        obs = policy_batch['observations']  / self.rescale
        obs = torch.index_select(obs, 1, self.state_indices)

        if self.wrap_absorbing:
            pass
            # obs = torch.cat([obs, policy_batch['absorbing'][:, 0:1]], dim=-1)
        else:
            self.ebm.eval()
            ebm_input = obs
           
        ebm_rew = self.get_energy(ebm_input).detach()
        
        if self.clamp_magnitude > 0:
            print(self.clamp_magnitude, self.clamp_magnitude is None)
            ebm_rew = torch.clamp(ebm_rew, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)

        # compute the reward using the algorithm
        shape_reward = torch.index_select(obs, 1, torch.LongTensor([0]).to(ptu.device)) # torch.index_select(policy_batch['actions'], 1, torch.LongTensor([0]).to(ptu.device))
        policy_batch['rewards'] = reward_func(self.rew_func, self.cons)(ebm_rew) * shape_reward

        # print('ebm_obs: ', np.mean(ptu.get_numpy(policy_batch['observations']), axis=0))
        # print('ebm: ', np.mean(ptu.get_numpy(ebm_rew)))
        # print('ebm_rew: ', np.mean(ptu.get_numpy(policy_batch['rewards'])))

        # buffer = self.target_state_buffer
        # exp_obs = buffer[np.random.choice(buffer.shape[0], size=100)]
        # exp_input = torch.Tensor(exp_obs / self.rescale).to(ptu.device)
        # exp_ebm = self.get_energy(exp_input).detach()
        # exp_rew = reward_func(self.rew_func, self.cons)(exp_ebm)

        # print('exp_obs: ', np.mean(exp_obs,axis=0))
        # print("exp data ebm: ", np.mean(ptu.get_numpy(exp_ebm)))
        # print("exp data rew: ", np.mean(ptu.get_numpy(exp_rew)))
        
        # policy optimization step
        self.policy_trainer.train_step(policy_batch)

        """
        Save some statistics for eval
        """
        if self.ebm_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.ebm_eval_statistics = OrderedDict()

        self.ebm_eval_statistics['ebm Rew Mean'] = np.mean(ptu.get_numpy(policy_batch['rewards']))
        self.ebm_eval_statistics['ebm Rew Std'] = np.std(ptu.get_numpy(policy_batch['rewards']))
        self.ebm_eval_statistics['ebm Rew Max'] = np.max(ptu.get_numpy(policy_batch['rewards']))
        self.ebm_eval_statistics['ebm Rew Min'] = np.min(ptu.get_numpy(policy_batch['rewards']))
    
    
    @property
    def networks(self):
        return [self.ebm, self.exploration_policy] + self.policy_trainer.networks


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot


    def to(self, device):
        self.state_indices = self.state_indices.to(ptu.device)
        super().to(device)
