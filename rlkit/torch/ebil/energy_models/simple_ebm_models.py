import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch import pytorch_util as ptu


class MLPEBM(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act='relu',
        use_bn=True,
        clamp_magnitude=None,
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude

        self.mod_list = nn.ModuleList([nn.Linear(input_dim, hid_dim)])
        if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
        self.mod_list.append(hid_act_class())

        for i in range(num_layer_blocks - 1):
            self.mod_list.append(nn.Linear(hid_dim, hid_dim))
            # if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
            self.mod_list.append(hid_act_class())
        
        self.mod_list.append(nn.Linear(hid_dim, 1))
        self.mod_list.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.mod_list)


    def forward(self, batch):
        output = self.model(batch)
        # if self.clamp_magnitude is not None:
            # output = torch.clamp(output, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return output
