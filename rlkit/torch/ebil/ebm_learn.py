import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim

import gtimer as gt

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.core import logger, eval_util


class EBMLearn(TorchBaseAlgorithm):
    def __init__(
        self,

        mode, # 'deen'
        expert_replay_buffer,
        ebm,
        input_dim,
        eval_every_epoch,

        num_updates_per_train_call=-1,
        batch_size=256,
        best_key='random_expert_diff', # higher is better
        sigma=0.1,
        state_only=False,

        lr=1e-2,
        momentum=0.0,
        optimizer_class=optim.Adam,

        **kwargs
    ):
        assert mode in ['deen', 'ae'], 'Invalid mode!'
        if kwargs['wrap_absorbing']: raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.expert_replay_buffer = expert_replay_buffer
        self._loss = None

        self.eval_every_epoch = eval_every_epoch

        self.batch_size = batch_size
        self.ebm = ebm
        self.sigma = sigma
        self.input_dim = input_dim

        self.optimizer = optimizer_class(
            self.ebm.parameters(),
            lr=lr,
            betas=(momentum, 0.999)
        )
        
        if num_updates_per_train_call < 0:
            self.num_updates_per_train_call = int(self.expert_replay_buffer.num_steps_can_sample() / self.batch_size)
        else:
            self.num_updates_per_train_call = num_updates_per_train_call
        
        # self.num_updates_per_train_call = 1
        # self.input_dim = 2

        self.best_key = best_key
        self.best_epoch = -1
        self.best_random_avg_energy = -999
        self.best_expert_avg_energy = 999


    def get_batch(self, batch_size, keys=None, use_expert_buffer=True):
        if use_expert_buffer:
            rb = self.expert_replay_buffer
        else:
            rb = self.replay_buffer
        batch = rb.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch


    def _do_training(self, epoch):
        for t in range(self.num_updates_per_train_call):
            self._do_update_step(epoch, use_expert_buffer=True)


    def _do_update_step(self, epoch, use_expert_buffer=True):
        batch = self.get_batch(
            self.batch_size,
            keys=['observations', 'actions'],
            use_expert_buffer=use_expert_buffer
        )

        # batch = {'observations':torch.Tensor([[0.],[1.],[2.],[3.],[4.],[5.],[6.],[7.],[8.],[9.],[10.]]), 'actions':torch.Tensor([[0.5]]*11)}
        
        obs = batch['observations']
        acts = batch['actions']

        if len(acts.shape) < 2:
            acts = torch.unsqueeze(acts, 1)

        self.optimizer.zero_grad()
        if self.mode == 'deen':
            batch_data = torch.cat([obs, acts], dim=1).to(ptu.device)
            self.noise = torch.normal(torch.zeros(batch_data.shape), self.sigma*torch.ones(batch_data.shape)).to(ptu.device)
            noise_data = batch_data + self.noise
            noise_data.requires_grad=True
            self.energy = self.ebm(noise_data).sum()
           
            E_y_gradient = torch.autograd.grad(self.energy, noise_data, create_graph=True)[0] # is a tuple
            SigmaSquare_E_y_gradient = self.sigma**2 * E_y_gradient
            
            self._loss = loss = ((batch_data - noise_data + SigmaSquare_E_y_gradient)**2).mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics['Loss'] = ptu.get_numpy(loss)
        elif self.mode == 'ae':
            batch_data = torch.cat([obs, acts], dim=1).to(ptu.device)
            predict_data = self.ebm(batch_data)
            
            self._loss = loss = ((batch_data - predict_data)**2).mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics['Loss'] = ptu.get_numpy(loss)
        loss.backward()
        self.optimizer.step()

    def train(self, start_epoch=0):
        self.pretrain()
        self.training_mode(False)
        gt.reset()
        gt.set_def_unique(False)
        self.start_training(start_epoch=start_epoch)

    def start_training(self, start_epoch=0):
        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            gt.stamp('sample')
            self._try_to_train(epoch)
            gt.stamp('train')
            
            if epoch % self.eval_every_epoch == 0:
                gt.stamp('sample')
                self._try_to_eval(epoch)
                gt.stamp('eval')
            self._end_epoch()
    
    def _try_to_train(self, epoch):
        if self._can_train():
            self.training_mode(True)
            self._do_training(epoch)
            self._n_train_steps_total += 1
            self.training_mode(False)

    def _try_to_eval(self, epoch):
        if self._can_evaluate():
            # save if it's time to save
            if epoch % self.freq_saving == 0:
                logger.save_extra_data(self.get_extra_data_to_save(epoch))
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)

            self.evaluate(epoch)

            logger.record_tabular(
                "Number of train calls total",
                self._n_train_steps_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def get_energy(self, data):
        if self.mode == 'deen':
            return self.ebm(data)
        elif self.mode == 'ae':
            return (data - self.ebm(data))**2
        else:
            raise NotImplementedError


    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except:
            print('No Stats to Eval')

        logger.log("Collecting random samples for evaluation")
        
        eval_steps = self.num_steps_per_eval
        
        test_paths = self.eval_sampler.obtain_samples(eval_steps)
        obs = torch.Tensor(np.squeeze(np.vstack([path["observations"] for path in test_paths])))
        acts = torch.Tensor(np.squeeze(np.vstack([path["actions"] for path in test_paths])))
        if len(acts.shape) < 2:
            acts = torch.unsqueeze(acts, 1)
        random_input = torch.cat([obs, acts], dim=1).to(ptu.device)
        

        exp_batch = self.get_batch(
            eval_steps,
            keys=['observations', 'actions'],
            use_expert_buffer=True
        )
        # exp_batch = {'observations':torch.Tensor([[0.],[1.],[2.],[3.],[4.],[5.],[6.],[7.],[8.],[9.],[10.]]), 'actions':torch.Tensor([[0.5]]*11)}

        obs = exp_batch['observations']
        acts = exp_batch['actions']
        exp_input = torch.cat([obs, acts], dim=1).to(ptu.device)
        
        statistics['random_avg_energy'] = self.ebm(random_input).mean().item()
        statistics['expert_avg_energy'] = self.get_energy(exp_input).mean().item()
        statistics['expert*20_avg_energy'] = self.get_energy(exp_input*20).mean().item()

        statistics["random_expert_diff"] = statistics["random_avg_energy"] - statistics["expert_avg_energy"]

        for key, value in statistics.items():
            logger.record_tabular(key, value)
        
        best_statistic = statistics[self.best_key]
        
        if best_statistic > self.best_statistic_so_far:
            self.best_statistic_so_far = best_statistic
            self.best_epoch = epoch
            self.best_random_avg_energy = statistics['random_avg_energy']
            self.best_expert_avg_energy = statistics['expert_avg_energy']
            logger.record_tabular("Best Model Epoch", self.best_epoch)
            logger.record_tabular("Best Random Energy", self.best_random_avg_energy)
            logger.record_tabular("Best Expert Energy", self.best_expert_avg_energy)
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {
                    'epoch': epoch,
                    'statistics': statistics
                }
                data_to_save.update(self.get_epoch_snapshot(epoch))
                logger.save_extra_data(data_to_save, 'best.pkl')
                print('\n\nSAVED BEST\n\n')
        logger.record_tabular("Best Model Epoch", self.best_epoch)
        logger.record_tabular("Best Random Energy", self.best_random_avg_energy)
        logger.record_tabular("Best Expert Energy", self.best_expert_avg_energy)
        

    def get_epoch_snapshot(self, epoch):
        """
        Probably will be overridden by each algorithm
        """
        data_to_save = dict(
            epoch=epoch,
            ebm=self.ebm,
        )
        return data_to_save

    def _can_evaluate(self):
        return (
            self._n_train_steps_total > 0
        )
    
    def _can_train(self):
        return self.expert_replay_buffer.num_steps_can_sample() > 0

    @property
    def networks(self):
        return [self.ebm, self.exploration_policy]

    def to(self, device):
        super().to(device)