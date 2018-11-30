import numpy as np
import torch
import gym
import random
import math
import time

from src.models import MDPnet, QNet, TerminalClassifier
from src.config import cartpole_config
from src.memory import *
from src.utils import *
from src.train_pipeline import *
from collections import deque

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def tune_alpha_pipeline(env, config, eval_qnet, seedvec = None, alpha_list=[0.1]):
    memory = SampleSet(config)
    dev_memory = SampleSet(config)
    traj_set = TrajectorySet(config)
    scores = deque()
    mdpnet_list = []
    for a in alpha_list:
        config.alpha_rep = a
        mdpnet_list.append(MDPnet(config))
    time_pre = time.time()

    # fix initial state
    if seedvec is None:
        seedvec = np.random.randint(0, config.MAX_SEED, config.sample_num_traj)

    for i_episode in range(config.sample_num_traj):
        # Initialize the environment and state
        randseed = seedvec[i_episode].item()
        env.seed(randseed)
        state = preprocess_state(env.reset(), config.state_dim)
        done = False
        n_steps = 0
        acc_soft_isweight = FloatTensor([1])
        acc_isweight = FloatTensor([1])
        factual = 1
        last_factual = 1
        traj_set.new_traj()
        while not done:
            # Select and perform an action
            q_values = eval_qnet.forward(Variable(state, volatile=True).type(FloatTensor)).data
            action = epsilon_greedy_action(state, eval_qnet, config.behavior_epsilon, config.action_size, q_values)
            p_pib = epsilon_greedy_action_prob(state, eval_qnet, config.behavior_epsilon,
                                               config.action_size, q_values)
            soft_pie = epsilon_greedy_action_prob(state, eval_qnet, config.soften_epsilon,
                                               config.action_size, q_values)
            p_pie = epsilon_greedy_action_prob(state, eval_qnet, 0, config.action_size, q_values)
            # action, p_pib = soft_policy(q_values, alpha=0.8, beta=0.05)
            # pie_action, p_pie = soft_policy(q_values, alpha=0.9, beta=0.05)
            # soft_action, soft_pie = soft_policy(q_values, alpha=0.9, beta=0.05)

            isweight = p_pie[:, action[0, 0]] / p_pib[:, action[0, 0]]
            acc_isweight = acc_isweight * (p_pie[:, action[0, 0]] / p_pib[:, action[0, 0]])
            soft_isweight = (soft_pie[:, action[0, 0]] / p_pib[:, action[0, 0]])
            acc_soft_isweight = acc_soft_isweight * (soft_pie[:, action[0, 0]] / p_pib[:, action[0, 0]])

            last_factual = factual * (1 - p_pie[:, action[0, 0]]) # 1{a_{0:t-1}==\pie, a_t != \pie}
            factual = factual * p_pie[:, action[0, 0]] # 1{a_{0:t}==\pie}

            next_state, reward, done, _ = env.step(action[0, 0])
            reward = Tensor([reward])
            next_state = preprocess_state(next_state, config.state_dim)
            if i_episode < config.train_num_traj:
                memory.push(state, action, next_state, reward, done, isweight, acc_isweight, n_steps, factual,
                            last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)
            else:
                dev_memory.push(state, action, next_state, reward, done, isweight, acc_isweight, n_steps, factual,
                                last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)
            traj_set.push(state, action, next_state, reward, done, isweight, acc_isweight, n_steps, factual,
                          last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)
            state = FloatTensor(next_state)
            n_steps += 1
        scores.append(n_steps)
    memory.flatten() # prepare flatten data
    dev_memory.flatten()
    memory.update_u() # prepare u_{0:t}
    dev_memory.update_u()
    mean_score = np.mean(scores)
    print('Sampling {} trajectories, the mean survival time is {}'
          .format(config.sample_num_traj, mean_score))
    print('{} train samples, {} dev sample'.format(len(memory), len(dev_memory)))
    time_now = time.time()
    time_sampling = time_now-time_pre
    time_pre = time_now

    for i, a in enumerate(alpha_list):
        print('Learn our mdp model with alpha {}'.format(a))
        config.alpha_rep = a
        best_train_loss = 100
        lr = config.lr
        for i_episode in range(config.train_num_episodes):
            train_loss = 0
            dev_loss = 0
            optimizer = optim.Adam(mdpnet_list[i].parameters(), lr=lr)
            for i_batch in range(config.train_num_batches):
                train_loss_batch = mdpmodel_train(memory, mdpnet_list[i], optimizer, 1, config)
                dev_loss_batch = mdpmodel_test(dev_memory, mdpnet_list[i], 1, config)
                train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
                dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
            if (i_episode + 1) % config.print_per_epi == 0:
                print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}'
                      .format(i_episode + 1, train_loss, dev_loss))
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            else:
                lr *= config.lr_decay
    time_now = time.time()
    time_mdp = time_now - time_pre
    time_pre = time_now

    tc = TerminalClassifier(config)
    print('Learn terminal classifier')
    lr = 0.01
    best_train_acc = 0
    optimizer = optim.Adam([param for param in tc.parameters() if param.requires_grad], lr=lr)
    for i_episode in range(config.tc_num_episode):
        train_loss = 0
        dev_loss = 0
        train_acc = 0
        dev_acc = 0
        for i_batch in range(config.tc_num_batches):
            train_loss_batch, train_acc_batch = terminal_classifier_train(memory, tc, optimizer,
                                                                          config.tc_batch_size)
            dev_loss_batch, dev_acc_batch = terminal_classifier_test(dev_memory, tc,
                                                                     config.test_batch_size)
            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            train_acc = (train_acc * i_batch + train_acc_batch) / (i_batch + 1)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
            dev_acc = (dev_acc * i_batch + dev_acc_batch) / (i_batch + 1)
        if (i_episode+1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e} acc {:.3e}, dev loss {:.3e} acc {:.3e}'.
                  format(i_episode + 1, train_loss, train_acc, dev_loss, dev_acc))
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        else:
            lr *= 0.9
            learning_rate_update(optimizer, lr)
    time_now = time.time()
    time_tc = time_now - time_pre
    time_pre = time_now

    print('Evaluate models using evaluation policy on the same initial states')
    for i, a in enumerate(alpha_list):
        mdpnet_list[i].eval()
    target = np.zeros(config.sample_num_traj)

    init_states = []
    for i_episode in range(config.sample_num_traj):
        env.seed(seedvec[i_episode].item())
        init_states.append(preprocess_state(env.reset(), config.state_dim))
    init_states = torch.cat(init_states)
    estm = []
    for i, a in enumerate(alpha_list):
        estm.append(rollout_batch(init_states, mdpnet_list[i], tc, config.eval_num_rollout, eval_qnet,
                      epsilon=0, action_size=config.action_size, maxlength=config.max_length))
    time_now = time.time()
    time_eval = time_now - time_pre
    time_pre = time_now

    # dr = doubly_robust(traj_set, mdpnet, tc, eval_qnet, config)
    time_now = time.time()
    time_dr = time_now - time_pre
    time_pre = time_now

    for i_episode in range(config.sample_num_traj):
        env.seed(seedvec[i_episode].item())
        state = preprocess_state(env.reset(), config.state_dim)
        true_state = state
        true_done = False
        true_steps = 0
        while not true_done:
            true_action = select_maxq_action(true_state, eval_qnet)
            true_next_state, true_reward, true_done, _ = env.step(true_action[0, 0])
            true_state = preprocess_state(true_next_state, config.state_dim)
            true_steps += 1

        target[i_episode] = true_steps
    time_now = time.time()
    time_gt = time_now - time_pre

    print('| Sampling: {:.3f}s | Learn MDP: {:.3f}s | Learn Termination: {:.3f}s '
          '| Evaluation: {:.3f}s | DR: {:.3f}s | Groudtruth: {:.3f}s |'
          .format(time_sampling,time_mdp,time_tc,time_eval,time_dr,time_gt))
    return estm, target


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    config = cartpole_config
    eval_qnet = QNet(config.state_dim,config.dqn_hidden_dims,config.action_size)
    load_qnet(eval_qnet,filename='qnet_cp_long.pth.tar')
    eval_qnet.eval()

    mse = []
    ind_mse = []
    alpha_list = [0, 0.01, 0.1, 1.0, 10]
    for alpha in alpha_list:
        mse.append(deque())
        ind_mse.append(deque())

    for i in range(config.N):
        print('Run: {}'.format(i+1))
        seedvec = np.random.randint(0, config.MAX_SEED, config.sample_num_traj)
        estm, target = tune_alpha_pipeline(env, config, eval_qnet, seedvec, alpha_list)
        for i, alpha in enumerate(alpha_list):
            mse_1, mse_2 = error_info(estm[i], target, 'Model   ')
            mse[i].append(mse_1)
            ind_mse[i].append(mse_2)

    mse_table = np.zeros((len(alpha_list),4))
    print('Average result over {} runs:'.format(config.N))
    for i in range(len(alpha_list)):
        print('Model (alpha={}): Root mse of mean is {:.3e}±{:.2e}, root mse of individual is {:.3e}±{:.2e}'
              .format(alpha, np.sqrt(np.mean(mse[i])), np.sqrt(np.std(mse[i])),
                      np.sqrt(np.mean(ind_mse[i])), np.sqrt(np.std(ind_mse[i]))))
        mse_table[i, 0] = np.sqrt(np.mean(mse[i]))
        mse_table[i, 1] = np.sqrt(np.std(mse[i]))
        mse_table[i, 2] = np.sqrt(np.mean(ind_mse[i]))
        mse_table[i, 3] = np.sqrt(np.std(ind_mse[i]))
    np.savetxt('results/result_cartpole_alphas.txt', mse_table, fmt='%.3e', delimiter=',')

