import numpy as np
import torch
from torch import nn


def rl_to_evo(rl_net, evo_net):
    for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
        target_param.data.copy_(param.data)

def re_reward(l):
    gp = np.max(l) - np.min(l)
    l = l - np.min(l)
    l = l + gp * 0.001
    return l

def define_loss(loss_f):
    if loss_f == 'L1':
        lossfunc = nn.L1Loss()
    elif loss_f == 'MSE':
        lossfunc = nn.MSELoss()
    elif loss_f == 'Entropy':
        lossfunc = nn.CrossEntropyLoss()
    else:
        raise ("args loss error! L1 or MSE or Entropy! ")
    return lossfunc

