import numpy as np
import random
import torch
import time
import inspect
import gym
import os
from torch import nn, optim
from torch.multiprocessing import Process

def build_fnn(input_size,output_size,n_layers,hidden_size,activation=nn.ReLU,out_activation=None):

    layers = []

    for _ in range(n_layers):
        layers += [nn.Linear(input_size, hidden_size), activation()]
        input_size = hidden_size
    if out_activation is None:
        layers += [nn.Linear(hidden_size, output_size)]
    else:
        layers += [nn.Linear(hidden_size, output_size), out_activation()]

    return nn.Sequential(*layers).apply(weight_init)

def weight_init(m):
    if hasattr(m, 'weight'):
        torch.nn.init.xavier_uniform_(m.weight)

def setup_logger(log_dir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(log_dir)
    # Log experimental parameters
    args = inspect.getargspec(train)[0]
    hyper_params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_hyperparams(hyper_params)

def define_loss(loss_f):
    if loss_f=='MSE':
        return nn.MSELoss()
    else:
        print('loss function name error:(\'MSE\'=nn.MSELoss())')
        raise NameError

class QNet(nn.Module):
    def __init__(self, q_net_args):
        super(QNet, self).__init__()
        self.ob_dim = q_net_args['ob_dim']
        self.ac_dim = q_net_args['ac_dim']
        self.hidden_size = q_net_args['hidden_size']
        self.n_layers = q_net_args['n_layers']

        self.fnn = build_fnn(self.ob_dim + self.ac_dim, 1, self.n_layers, self.hidden_size)


    def forward(self, t_x):
        # t_x = (batch_size, ob_dim+ac_dim)
        return self.fnn(t_x)


class PolicyNet(nn.Module):
    def __init__(self, pi_net_args, use_cuda=True):
        super(PolicyNet, self).__init__()
        self.ob_dim = pi_net_args['ob_dim']
        self.ac_dim = pi_net_args['ac_dim']
        self.hidden_size = pi_net_args['hidden_size']
        self.n_layers = pi_net_args['n_layers']
        self.device = pi_net_args['device']
        self.fnn = build_fnn(self.ob_dim, self.ac_dim, self.n_layers, self.hidden_size, out_activation=nn.Tanh)

        self.std = nn.Parameter(torch.randn((self.ac_dim,)))




    def forward(self, t_x):
        t_mean = self.fnn(t_x)
        t_std = self.std
        return t_mean, t_std

    def get_action(self, t_s):
        t_mean, t_std = self.forward(t_s)
        t_ac = torch.normal(mean=t_mean, std=t_std.exp())
        return t_mean.detach().cpu().numpy()     # maybe dim problem

    def get_action_gpu(self, t_s):
        t_mean, _ = self.forward(t_s)
        return t_mean     # maybe dim problem

    # def get_action_tensor(self, t_s):
    #     t_mean, t_std = self.forward(t_s)
    #     t_ac = torch.normal(mean=t_mean, std=t_std.exp())
    #     return t_mean     # maybe dim problem

    def sample_action(self, t_s):
        t_s = torch.from_numpy(t_s).float().cuda()
        #print(type(t_s))
        t_a = self.get_action(t_s)
        return t_a

    def get_batch_action(self, t_s):
        t_mean, t_std = self.forward(t_s)
        t_pad = torch.zeros_like(t_mean)
        t_pad[:] = t_std
        t_ac = torch.normal(mean=t_mean, std=t_pad.exp())
        return t_mean


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position +1)%self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)




class Agent:
    def __init__(self, q_net_args, pi_net_args, learn_args):
        self.q_net = QNet(q_net_args)
        self.target_q_net = QNet(q_net_args)
        self.policy_net = PolicyNet(pi_net_args)
        self.target_policy_net = PolicyNet(pi_net_args)
        self.imitation_net = PolicyNet(pi_net_args)

        self.batch_size = learn_args['batch_size']
        self.gamma = learn_args['gamma']
        self.soft_tau = learn_args['soft_tau']
        self.buffer_size = learn_args['buffer_size']
        self.max_frame = learn_args['max_frame']
        self.max_step = learn_args['max_step']
        self.q_lr = learn_args['q_lr']
        self.pi_lr = learn_args['pi_lr']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_q_value = learn_args['max_q_value']
        self.min_q_value = learn_args['min_q_value']
        self.loss_function = define_loss(learn_args['loss_function'])

        q_params = list(self.q_net.parameters())
        pi_params = list(self.policy_net.parameters())

        self.q_optimizer = optim.Adam(q_params, lr=self.q_lr, weight_decay=learn_args['L2WeightDecay'])
        self.pi_optimizer = optim.Adam(pi_params, lr=self.pi_lr)

        self.buffer = ReplayBuffer(self.buffer_size)
        for target_params, params in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_params.data.copy_(params.data)

        for target_params, params in zip(self.imitation_net.parameters(), self.policy_net.parameters()):
            target_params.data.copy_(params.data)

        for target_params, params in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_params.data.copy_(params.data)

        self.policy_net.to(self.device)
        self.q_net.to(self.device)
        self.target_policy_net.to(self.device)
        self.target_q_net.to(self.device)
        self.imitation_net.to(self.device)

    def sample_action(self, t_s):
        t_s = torch.from_numpy(t_s).float().cuda()
        t_a = self.policy_net.get_action(t_s)
        return t_a

    def sample_ac_with_IL(self, t_s):
        t_s = torch.from_numpy(t_s).float().cuda()
        t_a_pi = self.policy_net.get_action_tensor(t_s)
        # t_a_pi = torch.from_numpy(t_a_pi).float().cuda()
        # print(t_s)
        # print(t_a_pi)
        # print(torch.cat([t_s, t_a_pi], 0))


        q_s_a_pi = self.q_net(torch.cat([t_s, t_a_pi], 0))
        t_a_IL = self.imitation_net.get_action_tensor(t_s)
        # t_a_IL = torch.from_numpy(t_a_IL).float().cuda()
        q_s_a_IL = self.q_net(torch.cat([t_s, t_a_IL], 0))
        if q_s_a_IL > q_s_a_pi:
            return t_a_IL.detach().cpu().numpy()
        else:
            return t_a_pi.detach().cpu().numpy()

    def get_batch_action(self, t_s):
        t_a_pi = self.policy_net.get_action_gpu(t_s)
        q_s_a_pi = self.q_net(torch.cat([t_s, t_a_pi], 1))
        # t_a_pi = t_a_pi.detach().cpu().numpy()
        # q_s_a_pi = q_s_a_pi.detach().cpu().numpy()

        t_a_IL = self.imitation_net.get_action_gpu(t_s)
        q_s_a_IL = self.q_net(torch.cat([t_s, t_a_IL], 1))
        # t_a_IL = t_a_IL.detach().cpu().numpy()
        # q_s_a_IL = q_s_a_IL.detach().cpu().numpy()

        a = torch.zeros(t_a_pi.shape)
        for i in range(len(q_s_a_pi)):
            if q_s_a_pi[0] > q_s_a_IL[0]:
                a[i] = t_a_pi[i]
            else:
                a[i] = t_a_IL[i]
        # a = torch.FloatTensor(a).to(self.device)
        a = a.to(self.device)
        return a

    def get_pg_batch_action(self, t_s):
        t_a_pi = self.policy_net.get_batch_action(t_s)
        return t_a_pi

    def get_ea_batch_action(self, t_s):
        t_a_IL = self.imitation_net.get_batch_action(t_s)
        return t_a_IL

    def update_with_IL_pg(self):
        t_s, t_a, t_r, t_ns, t_d = self.buffer.sample(self.batch_size)

        t_s = torch.FloatTensor(t_s).to(self.device)
        t_ns = torch.FloatTensor(t_ns).to(self.device)
        t_a = torch.FloatTensor(t_a).to(self.device)
        t_r = torch.FloatTensor(t_r).unsqueeze(1).to(self.device)
        t_d = torch.FloatTensor(np.float32(t_d)).unsqueeze(1).to(self.device)

        pi_loss = self.q_net(torch.cat([t_s, self.policy_net.get_batch_action(t_s)], 1))
        pi_loss = -pi_loss.mean()


        t_nv = self.target_q_net(torch.cat([t_ns, self.get_pg_batch_action(t_ns).detach()], 1))
        t_y = t_r + (1.0 - t_d) * self.gamma * t_nv
        t_y = torch.clamp(t_y, self.min_q_value, self.max_q_value)

        t_q = self.q_net(torch.cat([t_s, t_a], 1))
        q_loss = self.loss_function(t_q, t_y.detach())

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        for target_params, params in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_params.data.copy_(
                self.soft_tau * params.data + (1.0 - self.soft_tau) * target_params.data
            )

        for target_params, params in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_params.data.copy_(
                self.soft_tau * params.data + (1.0 - self.soft_tau) * target_params.data
            )

    # def update_with_IL_ea(self):
    #     t_s, t_a, t_r, t_ns, t_d = self.buffer.sample(self.batch_size)
    #
    #     t_s = torch.FloatTensor(t_s).to(self.device)
    #     t_ns = torch.FloatTensor(t_ns).to(self.device)
    #     t_a = torch.FloatTensor(t_a).to(self.device)
    #     t_r = torch.FloatTensor(t_r).unsqueeze(1).to(self.device)
    #     t_d = torch.FloatTensor(np.float32(t_d)).unsqueeze(1).to(self.device)
    #
    #     pi_loss = self.q_net(torch.cat([t_s, self.policy_net.get_batch_action(t_s)], 1))
    #     pi_loss = -pi_loss.mean()
    #
    #
    #     t_nv = self.target_q_net(torch.cat([t_ns, self.get_ea_batch_action(t_ns).detach()], 1))
    #     t_y = t_r + (1.0 - t_d) * self.gamma * t_nv
    #     t_y = torch.clamp(t_y, self.min_q_value, self.max_q_value)
    #
    #     t_q = self.q_net(torch.cat([t_s, t_a], 1))
    #     q_loss = self.loss_function(t_q, t_y.detach())
    #
    #     self.pi_optimizer.zero_grad()
    #     pi_loss.backward()
    #     self.pi_optimizer.step()
    #
    #     self.q_optimizer.zero_grad()
    #     q_loss.backward()
    #     self.q_optimizer.step()
    #
    #     for target_params, params in zip(self.target_q_net.parameters(), self.q_net.parameters()):
    #         target_params.data.copy_(
    #             self.soft_tau * params.data + (1.0 - self.soft_tau) * target_params.data
    #         )
    #
    #     for target_params, params in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
    #         target_params.data.copy_(
    #             self.soft_tau * params.data + (1.0 - self.soft_tau) * target_params.data
    #         )


    def update_with_IL(self):
        t_s, t_a, t_r, t_ns, t_d = self.buffer.sample(self.batch_size)

        t_s = torch.FloatTensor(t_s).to(self.device)
        t_ns = torch.FloatTensor(t_ns).to(self.device)
        t_a = torch.FloatTensor(t_a).to(self.device)
        t_r = torch.FloatTensor(t_r).unsqueeze(1).to(self.device)
        t_d = torch.FloatTensor(np.float32(t_d)).unsqueeze(1).to(self.device)

        pi_loss = self.q_net(torch.cat([t_s, self.policy_net.get_batch_action(t_s)], 1))
        pi_loss = -pi_loss.mean()

        a = self.get_batch_action(t_ns).detach()
        # print(type(a))

        t_nv = self.target_q_net(torch.cat([t_ns, a], 1))
        t_y = t_r + (1.0 - t_d) * self.gamma * t_nv
        t_y = torch.clamp(t_y, self.min_q_value, self.max_q_value)

        t_q = self.q_net(torch.cat([t_s, t_a], 1))
        q_loss = self.loss_function(t_q, t_y.detach())

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        for target_params, params in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_params.data.copy_(
                self.soft_tau * params.data + (1.0 - self.soft_tau) * target_params.data
            )

        for target_params, params in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_params.data.copy_(
                self.soft_tau * params.data + (1.0 - self.soft_tau) * target_params.data
            )


    def update(self):
        t_s, t_a, t_r, t_ns, t_d = self.buffer.sample(self.batch_size)

        t_s = torch.FloatTensor(t_s).to(self.device)
        t_ns = torch.FloatTensor(t_ns).to(self.device)
        t_a = torch.FloatTensor(t_a).to(self.device)
        t_r = torch.FloatTensor(t_r).unsqueeze(1).to(self.device)
        t_d = torch.FloatTensor(np.float32(t_d)).unsqueeze(1).to(self.device)

        pi_loss = self.q_net(torch.cat([t_s, self.policy_net.get_batch_action(t_s)], 1))
        pi_loss = -pi_loss.mean()

        t_nv = self.target_q_net(torch.cat([t_ns, self.target_policy_net.get_batch_action(t_ns).detach()], 1))
        t_y = t_r + (1.0 - t_d) * self.gamma * t_nv
        t_y = torch.clamp(t_y, self.min_q_value, self.max_q_value)

        t_q = self.q_net(torch.cat([t_s, t_a], 1))
        q_loss = self.loss_function(t_q, t_y.detach())

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        for target_params, params in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_params.data.copy_(
                self.soft_tau * params.data + (1.0 - self.soft_tau) * target_params.data
            )

        for target_params, params in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_params.data.copy_(
                self.soft_tau * params.data + (1.0 - self.soft_tau) * target_params.data
            )






