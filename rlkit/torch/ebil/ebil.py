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

def reward_func(args):
    if args.reward_func == '-energy':
        return lambda x:-x
    elif args.reward_func == '-energy+1':
        return lambda x:1-x
    elif args.reward_func == '-energy-1':
        return lambda x:-1-x
    elif args.reward_func == 'exp(-energy-1)':
        return lambda x:tf.exp(-x-1)
    elif args.reward_func == '(-energy+1)div2':
        return lambda x:(-x+1)/2.0
    elif args.reward_func == 'exp(-energy+1)':
        return lambda x:tf.exp(-x+1)
    elif args.reward_func == 'exp(-energy)':
        return lambda x:tf.exp(-x)
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

        expert_replay_buffer,

        state_only=False,

        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,

        num_policy_update_loops_per_train_call=1,
        num_policy_updates_per_loop_iter=100,

        use_grad_pen=True,
        grad_pen_weight=10,

        rew_clip_min=None,
        rew_clip_max=None,

        **kwargs
    ):
        assert mode in ['deen'], 'Invalid ebil algorithm!'
        if kwargs['wrap_absorbing']: raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.rew_func = rew_func
        self.state_only = state_only

        self.expert_replay_buffer = expert_replay_buffer

        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert
        
        self.ebm = ebm
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.num_policy_update_loops_per_train_call = num_policy_update_loops_per_train_call
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None


    def get_batch(self, batch_size, from_expert, keys=None):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch


    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        super()._end_epoch()


    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        super().evaluate(epoch)


    def _do_training(self, epoch, mode):
        for t in range(self.num_update_loops_per_train_call):
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
        
        obs = policy_batch['observations']
        acts = policy_batch['actions']
        next_obs = policy_batch['next_observations']

        if self.wrap_absorbing:
            pass
            # obs = torch.cat([obs, policy_batch['absorbing'][:, 0:1]], dim=-1)
        else:
            self.ebm.eval()
            if self.state_only:
                ebm_input = torch.cat([obs, next_obs], dim=1)
            else:
                ebm_input = torch.cat([obs, acts], dim=1)
            ebm_logits = self.ebm(ebm_input).detach()

        # compute the reward using the algorithm
        if self.mode == 'deen':
            policy_batch['rewards'] = ebm_logits
        else
            raise NotImplementedError
        
        if self.clip_max_rews:
            policy_batch['rewards'] = torch.clamp(policy_batch['rewards'], max=self.rew_clip_max)
        if self.clip_min_rews:
            policy_batch['rewards'] = torch.clamp(policy_batch['rewards'], min=self.rew_clip_min)

        policy_batch['rewards'] = reward_func(self.rew_func)
        
        # policy optimization step
        self.policy_trainer.train_step(policy_batch)

        self.ebm_eval_statistics['ebm Rew Mean'] = np.mean(ptu.get_numpy(policy_batch['rewards']))
        self.ebm_eval_statistics['ebm Rew Std'] = np.std(ptu.get_numpy(policy_batch['rewards']))
        self.ebm_eval_statistics['ebm Rew Max'] = np.max(ptu.get_numpy(policy_batch['rewards']))
        self.ebm_eval_statistics['ebm Rew Min'] = np.min(ptu.get_numpy(policy_batch['rewards']))
    
    
    @property
    def networks(self):
        return [self.ebm] + self.policy_trainer.networks


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot


    def to(self, device):
        super().to(device)
