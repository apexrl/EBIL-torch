import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm
from torch import autograd
from rlkit.core import logger


class BC(TorchBaseAlgorithm):
    def __init__(
        self,

        mode, # 'MLE' or 'MSE'
        expert_replay_buffer,
        critic,

        num_update_loops_per_train_call=1,
        num_pretrain_updates = 50,
        num_bc_updates_per_loop_iter=100,
        num_adp_updates_per_loop_iter=100,

        batch_size=1024,
        lr=1e-3,
        momentum=0.0,
        optimizer_class=optim.Adam,

        critic_optim_batch_size=256,
        critic_lr=1e-3,
        critic_momentum=0.0,
        critic_optimizer_class=optim.Adam,
        grad_pen_weight=10,

        n_itr_critic=5,

        **kwargs
    ):
        assert mode in ['MLE', 'MSE'], 'Invalid mode!'
        if kwargs['wrap_absorbing']: raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.critic = critic
        self.expert_replay_buffer = expert_replay_buffer

        self.batch_size = batch_size

        self.critic_batch_size = critic_optim_batch_size
        
        self.optimizer = optimizer_class(
            self.exploration_policy.parameters(),
            lr=lr,
            betas=(momentum, 0.999)
        )

        # critic and wd

        self.critic_optimizer = optimizer_class(
            self.critic.parameters(),
            lr=critic_lr,
            betas=(critic_momentum, 0.999)
        )

        self.wd_optimizer = optimizer_class(
            self.exploration_policy.parameters(),
            lr=critic_lr,
            betas=(critic_momentum, 0.999)
        )

        self.grad_pen_weight = grad_pen_weight

        self.num_pretrain_updates = num_pretrain_updates
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_adp_updates_per_loop_iter = num_adp_updates_per_loop_iter
        self.num_bc_updates_per_loop_iter = num_bc_updates_per_loop_iter
        self.n_itr_critic = n_itr_critic


    def get_batch(self, batch_size, keys=None, use_expert_buffer=True):
        if use_expert_buffer:
            rb = self.expert_replay_buffer
        else:
            rb = self.replay_buffer
        batch = rb.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def compile_features(self, source_input, target_input):
        source_out = self.exploration_policy(source_input)[-1]
        target_out = self.exploration_policy(target_input)[-1]

        return source_out, target_out

    def pretrain(self):
        logger.log('Pretraining ...')
        for ep in range(self.num_pretrain_updates):
            for t in range(self.num_update_loops_per_train_call):
                self._do_update_step(ep, use_expert_buffer=True)

    def _do_training(self, epoch):
        # logger.log('Start training ...')
        for t in range(self.num_update_loops_per_train_call):
            for _ in range(self.num_bc_updates_per_loop_iter):
                self._do_update_step(epoch, use_expert_buffer=True)
            for _ in range(self.num_adp_updates_per_loop_iter):
                self._do_adp_step(epoch)

    def _do_adp_step(self, epoch):

        for i in range(self.n_itr_critic):

            sb = self.get_batch(self.critic_batch_size, keys=['observations'], use_expert_buffer=True)['observations']
            tb = self.get_batch(self.critic_batch_size, keys=['observations'], use_expert_buffer=False)['observations']

            h_s, h_t = self.compile_features(sb, tb)

            critic_source_out = self.critic(h_s)
            critic_target_out = self.critic(h_t)
            
            self.wd_loss = wd_loss = critic_source_out.mean() - critic_target_out.mean()

            alpha = torch.rand((self.critic_batch_size, 1)).to(ptu.device) # (batchsize, 1)
            differences = h_s - h_t  # (batch_size, feature_dim)
            
            interpolates = h_t + (alpha * differences)
            critic_interpolates_out = self.critic(interpolates)

            gradients = autograd.grad(
                    outputs=critic_interpolates_out.sum(),
                    inputs=[interpolates],
                    create_graph=True, retain_graph=True, only_inputs=True
                )
            total_grad = gradients[0]
                
            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()  
            
            critic_loss = -self.wd_loss + self.grad_pen_weight * gradient_penalty

            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
            self.eval_statistics['critic loss'] = ptu.get_numpy(critic_loss)
            self.eval_statistics['wd loss'] = ptu.get_numpy(wd_loss)
            
       
            if i - self.n_itr_critic == -1:
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                self.wd_optimizer.zero_grad()
                wd_loss.backward()
                self.wd_optimizer.step()

            else:
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


    def _do_update_step(self, epoch, use_expert_buffer=True):
        batch = self.get_batch(
            self.batch_size,
            keys=['observations', 'actions'],
            use_expert_buffer=use_expert_buffer
        )
        
        obs = batch['observations']
        acts = batch['actions']

        self.optimizer.zero_grad()
        if self.mode == 'MLE':
            log_prob = self.exploration_policy.get_log_prob(obs, acts)
            loss = -1.0 * log_prob.mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
            self.eval_statistics['Log-Likelihood'] = ptu.get_numpy(-1.0*loss)
        else:
            pred_acts = self.exploration_policy(obs)[0]
            squared_diff = (pred_acts - acts)**2
            loss = torch.sum(squared_diff, dim=-1).mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
            self.eval_statistics['MSE'] = ptu.get_numpy(loss)
        loss.backward()
        self.optimizer.step()

    @property
    def networks(self):
        return [self.critic, self.exploration_policy]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(critic=self.critic)
        return snapshot

    def to(self, device):
        super().to(device)
