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

# version 4 for RIM
# add cross exploration [solved]
# add fair evaluation [solved][cant use]
# add il selection record [unsolved]
# add true rl evaluation [unsolved]
# add some logz [unsolved]

# add controllable cross exploration


class ERLAgent:

    def __init__(self, q_net_args, pi_net_args, learn_args, env, seed, exp_id):
        self.evolution = evolution.SSNE(learn_args)

        self.pop_size = learn_args['pop_size']  # population size
        self.buffer_size = learn_args['buffer_size']
        self.eval_nums = learn_args['num_evals']
        self.batch_size = learn_args['batch_size']
        self.update_frc = learn_args['update_frc']
        self.synch_period = learn_args['synch_period']
        self.device = pi_net_args['device']
        self.imitation_epochs = learn_args['imitation_epochs']
        self.i_batch_size = learn_args['i_batch_size']
        self.imitation_lr = learn_args['imitation_lr']
        self.imitation_loss_f = define_loss(learn_args['imitation_loss_f'])
        self.imitation_frc = learn_args['imitation_frc']
        self.epoch_size = 30000
        self.alpha = learn_args['alpha']
        self.is_cross = learn_args['cross']
        self.env = env

        self.rewards = []
        # self.smooth = []
        self.episode_lengths = []
        self.seed = seed
        self.exp_id = exp_id


        self.champ_scores = 0
        self.champ_scores_list = []
        self.frame_list = []
        self.rl_score = 0
        self.episode_num = 0
        # self.il_score = 0

        self.population = []
        for _ in range(self.pop_size):
            self.population.append(tDDPG.PolicyNet(pi_net_args).to(self.device))

        for pi in self.population: pi.eval() 


        self.rl_agent = tDDPG.Agent(q_net_args, pi_net_args, learn_args)
        self.ou_noise = OUNoise(self.env.action_space)

        self.num_games = 0
        self.num_frames = 0
        self.gen_frames = None

        test_score_rl = 0
        for eval in range(5):
            test_score_rl += self.evaluate(self.rl_agent.policy_net, is_render=False, store_transition=False,
                                           is_rl_eval=False)

        test_score_rl /= 5
        self.init_rl_score = test_score_rl
        print('init rl score is {}'.format(self.init_rl_score))


    def evaluate(self, net, is_render, is_action_noise=False, store_transition=True, is_rl_eval=False, is_random = False):
        total_reward = 0.0
        episode_length = 0
        state = self.env.reset()
        self.ou_noise.reset()
        done = False
        step = 0

        while not done:
            if store_transition:
                self.gen_frames += 1
                self.num_frames += 1

            if is_render:
                self.env.render()

            if is_rl_eval: # if it is evaluating RL
                # version 4 update --------- undo
                if is_random:
                    action = self.env.action_space.sample()
                else:
                    action = self.rl_agent.sample_action(state)
            else: # if it is evaluating EA
                if is_random:
                    action = self.env.action_space.sample()
                else:
                    action = net.sample_action(state)

            if is_action_noise: # exploration noise
                action = self.ou_noise.get_action(action, step)

            n_s, r, done, _ = self.env.step(action)
            action = np.clip(action, -1, 1)
            if store_transition:
                self.rl_agent.buffer.push(state,action,r,n_s,done)
            state = n_s
            total_reward += r
            episode_length += 1
            step += 1

        if is_rl_eval and store_transition:
            self.episode_num += 1
            self.rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            self.champ_scores_list.append(self.champ_scores)
            self.frame_list.append(self.num_frames)
            print("-"*41)
            print("|Exp_id       \t\t| %8.3g\t|"%(self.exp_id))
            print("|RandomSeed   \t\t| %8.3g\t|"%(self.seed))
            print("|RewardOfRL   \t\t| %8.3g\t|"%(total_reward))
            print("|EpisodeSteps \t\t| %8.3g\t|"%(episode_length))
            print("|RewardOfChamp\t\t| %8.3g\t|"%(self.champ_scores))  # final performance
            print("|FrameSoFar   \t\t| %8.3g\t|"%(self.num_frames))    # total steps
            print("-"*41)
            # save_rl_model(self.rl_agent.policy_net)

        return total_reward


    def imitation_rl(self, pop, agent):
        pop.train()
        optimizer = optim.Adam(pop.parameters(), lr=self.imitation_lr)
        epoch = 0
        while epoch <= self.epoch_size:
            if len(self.rl_agent.buffer) > self.i_batch_size:
                t_s, t_a, t_r, t_ns, t_d = self.rl_agent.buffer.sample(self.i_batch_size)
                t_s = torch.FloatTensor(t_s).to(self.device)

                t_expert_a = agent.get_batch_action(t_s)
                t_learner_a =pop.get_batch_action(t_s)

                optimizer.zero_grad()

                loss = self.imitation_loss_f(t_expert_a, t_learner_a)
                loss.backward()
                optimizer.step()
            epoch += 1

    def ddaggerCE(self, pop, agent, all_fitness):              # the paper doesn't use this function
        pop.train()
        state = self.env.reset()
        epoch = 0
        prios = np.array(all_fitness)
        prios = re_reward(prios)
        probs = prios ** self.alpha
        probs /= probs.sum()
        optimizer = optim.Adam(pop.parameters(), lr=self.imitation_lr)
        while epoch <= self.epoch_size:
            done = False
            while not done:
                # choose an individual and make a decision
                indices = np.random.choice(len(prios), 1, p=probs)
                individual = self.population[indices[0]]
                action = individual.sample_action(state)
                n_s, r, done, _ = self.env.step(action)
                self.rl_agent.buffer.push(state, action, r, n_s, done)
                state = n_s
                epoch += 1
                # imitation learning
                if len(self.rl_agent.buffer) > self.i_batch_size:
                    t_s, t_a, t_r, t_ns, t_d = self.rl_agent.buffer.sample(self.i_batch_size)
                    t_s = torch.FloatTensor(t_s).to(self.device)

                    t_expert_a = agent.get_batch_action(t_s)
                    t_learner_a = pop.get_batch_action(t_s)

                    optimizer.zero_grad()

                    loss = self.imitation_loss_f(t_expert_a, t_learner_a)
                    loss.backward()
                    optimizer.step()





    def il_train_exp(self):
        self.gen_frames = 0

        all_fitness = []


        for net in self.population:
            fitness = 0.0
            for eval in range(self.eval_nums):
                fitness += self.evaluate(net, is_render=False, store_transition=True, is_action_noise=False, is_random=self.num_frames<10000)
            all_fitness.append(fitness / self.eval_nums)



        best_train_fitness = max(all_fitness)
        worst_index = all_fitness.index(min(all_fitness))

        champ_index = all_fitness.index(max(all_fitness))
        test_score = 0.0


        for eval in range(5):
            test_score += self.evaluate(self.population[champ_index], is_render=False, store_transition=False, is_random=self.num_frames<10000)

        test_score /= 5

        self.champ_scores = test_score

        elite_index = self.evolution.epoch(self.population, all_fitness)


        ###############################################   ddpg   ####################################################
        self.evaluate(self.rl_agent.policy_net, is_render=False, store_transition=True, is_rl_eval=True)
        if len(self.rl_agent.buffer) > self.batch_size*5:
            for _ in range(int(self.gen_frames * self.update_frc)):
                self.rl_agent.update_with_IL()
                # soft update
                for target_params, params in zip(self.rl_agent.imitation_net.parameters(),
                                                 self.population[champ_index].parameters()):
                    target_params.data.copy_(
                        self.rl_agent.soft_tau * params.data + (1.0 - self.rl_agent.soft_tau) * target_params.data
                    )

            if self.episode_num % self.synch_period == self.synch_period -1:

                if self.is_cross:
                    self.ddaggerCE(self.population[worst_index], self.rl_agent, all_fitness) # the paper doesn't use this function
                else:
                    self.imitation_rl(self.population[worst_index], self.rl_agent)

                for eval in range(5):
                    test_score += self.evaluate(self.population[worst_index], is_render=False, store_transition=False)
                test_score /= 5
                # self.il_score = test_score
                self.evolution.il_policy = worst_index

            else:

                rl_to_evo(self.rl_agent.policy_net, self.population[worst_index])
                self.evolution.rl_policy = worst_index




        return best_train_fitness, test_score, elite_index


