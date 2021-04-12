import tDDPG
import evolution
import os
import numpy as np
import inspect
import torch
import torch.optim as optim
import gym
from torch.multiprocessing import Process
from utils import rl_to_evo, re_reward, define_loss
from paction import NormalizedActions
from ounoise import OUNoise
from RIM import ERLAgent


def train(exp_name, env_name, seed, n_layers, q_hidden_size, pi_hidden_size, learn_args, exp_id):


    env = NormalizedActions(gym.make(env_name))

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    ou_noise = OUNoise(env.action_space)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    q_net_args = {
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'n_layers': n_layers,
        'hidden_size': q_hidden_size,
        'device': device
    }

    pi_net_args = {
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'n_layers': n_layers,
        'hidden_size': pi_hidden_size,
        'device': device
    }

    if learn_args['max_step'] == -1:
        learn_args['max_step'] = env.spec.max_episode_steps

    agent = ERLAgent(q_net_args, pi_net_args, learn_args, env, seed, exp_id)


    while agent.num_frames <= learn_args['max_frame']:
        best_train_fitness, test_score, elite_index = agent.il_train_exp()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='InvertedDoublePendulum-v2')
    parser.add_argument('--exp_name', type=str, default='1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--max_frame', type=int, default=1000000)
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--learn_start', type=int, default=-1)
    parser.add_argument('--policy_learning_rate', '-plr', type=float, default=5e-5)
    parser.add_argument('--q_learning_rate', '-qlr', type=float, default=5e-4)
    parser.add_argument('--soft_tau', type=float, default=0.001)
    parser.add_argument('--max_step', type=int, default=-1)
    parser.add_argument('--loss_function', type=str, default='MSE')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--q_hidden_size', type=int, default=300)
    parser.add_argument('--pi_hidden_size', type=int, default=400)
    parser.add_argument('--L2WeightDecay', type=float, default=0.01)
    parser.add_argument('--max_q_value', type=int, default=-1)
    parser.add_argument('--min_q_value', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=3)
    parser.add_argument('--update_frc', type=int, default=1)
    parser.add_argument('--pop_size', type=int, default=10)
    parser.add_argument('--synch_period', type=int, default=10)  # ant==2, hopper=2
    parser.add_argument('--crossover_prob', type=float, default=0.0)
    parser.add_argument('--mutation_prob', type=float, default=0.9)
    parser.add_argument('--e_fraction', type=float, default=0.1)  # hopper,ant == 0.3  reacher,walker2d == 0.2 default==0.1
    parser.add_argument('--num_evals', type=int, default=1) # hopper,reacher ==5, Walker2d==3 default == 1
    parser.add_argument('--imitation_epochs', type=int, default=30000)
    parser.add_argument('--i_batch_size', type=int, default=32)
    parser.add_argument('--imitation_lr', type=float, default=0.001)
    parser.add_argument('--imitation_loss_f', type=str, default='L1')
    parser.add_argument('--imitation_frc', type=int, default=50)

    parser.add_argument('--alpha', type=int, default=0.6)
    parser.add_argument('--cross', type=bool, default=False)
    parser.add_argument('--fair_e', type=bool, default=False)




    args = parser.parse_args()



    processes = []

    learn_start = args.learn_start if args.learn_start>0 else args.batch_size
    if args.max_q_value > args.min_q_value:
        max_q_value, min_q_value = args.max_q_value, args.min_q_value
    else:
        max_q_value, min_q_value = np.inf, -np.inf

    learn_args = {
        'batch_size':args.batch_size,
        'buffer_size':args.buffer_size,
        'gamma':args.discount,
        'soft_tau':args.soft_tau,
        'max_frame':args.max_frame,
        'max_step':args.max_step,
        'q_lr':args.q_learning_rate,
        'pi_lr':args.policy_learning_rate,
        'loss_function':args.loss_function,
        'start_frame':learn_start,
        'max_q_value':max_q_value,
        'min_q_value':min_q_value,
        'L2WeightDecay':args.L2WeightDecay,
        'update_frc':args.update_frc,
        'pop_size':args.pop_size,
        'synch_period':args.synch_period,
        'crossover_prob':args.crossover_prob,
        'mutation_prob':args.mutation_prob,
        'e_fraction':args.e_fraction,
        'num_evals':args.num_evals,
        'imitation_epochs': args.imitation_epochs,
        'i_batch_size': args.i_batch_size,
        'imitation_lr': args.imitation_lr,
        'imitation_loss_f': args.imitation_loss_f,
        'imitation_frc': args.imitation_frc,
        'alpha': args.alpha,
        'cross':args.cross,
        'fair_e':args.fair_e
    }

    exp_id = 0
    for e in range(args.n_experiments):
        exp_id += 1
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train(args.exp_name, args.env_name, seed, args.n_layers, args.q_hidden_size, args.pi_hidden_size, learn_args, exp_id)
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

