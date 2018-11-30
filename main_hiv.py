import argparse
import pickle
import torch.optim as optim

from time import time
from sklearn.externals import joblib
from collections import deque
from src.memory import *
from src.utils import *
from src.models import MDPnet
from src.config import hiv_config
from hiv_domain.fittedQiter import FittedQIteration

from src.train_pipeline import mdpmodel_train, mdpmodel_test

parser = argparse.ArgumentParser()
parser.add_argument("--file_num", type=int, default=1)
parser.add_argument("--N", type=int)
parser.add_argument("--train_num_traj", type=int)
parser.add_argument("--dev_num_traj", type=int, default=50)
args = parser.parse_args()


def generate_data(env, config, eval_env=None):
    t0 = time()
    memory = SampleSet(config)
    dev_memory = SampleSet(config)
    traj_set = TrajectorySet(config)
    scores = deque()
    if config.standardize_rewards:
        with open("hiv_domain/standardization_data.pkl", "rb") as fobj:
            standardization_data = pickle.load(fobj)
            reward_mean = standardization_data[config.ins]["reward_mean"]
            reward_std = standardization_data[config.ins]["reward_std"]
    for i_episode in range(config.sample_num_traj):
        if i_episode % 10 == 0:
            print("{} trajectories generated".format(i_episode))
        episode = env.run_episode(eps=config.behavior_eps, track=True)
        done = False
        n_steps = 0
        factual = 1
        traj_set.new_traj()
        #acc_isweight = FloatTensor([1])
        while not done:
            action = int(np.where(episode[n_steps][1] == 1)[0][0])
            if eval_env:
                p_pie = float(eval_env.policy(episode[n_steps][0], eps=0) == action)
            else:
                p_pie = float(env.policy(episode[n_steps][0], eps=0) == action)
            int_pib = float(env.policy(episode[n_steps][0], eps=0) == action)
            if int_pib == 1:
                p_pib = (1 - config.behavior_eps *
                         (config.action_size-1)/config.action_size)
            else:
                p_pib = config.behavior_eps / config.action_size
            p_pie = FloatTensor([p_pie])
            p_pib = FloatTensor([p_pib])
            isweight = p_pie / p_pib
            #acc_isweight = acc_isweight * (p_pie / p_pib)
            last_factual = factual * (1 - p_pie)
            factual = factual * p_pie
            state = preprocess_state(episode[n_steps][0], config.state_dim)
            next_state = preprocess_state(episode[n_steps][3], config.state_dim)
            reward = episode[n_steps][2]
            if config.standardize_rewards:
                reward = (reward - reward_mean) / reward_std
            reward = preprocess_reward(reward)
            action = preprocess_action(action)
            done = n_steps == len(episode) - 1
            if i_episode < config.train_num_traj:
                memory.push(state, action, next_state, reward, done,
                            isweight, None, n_steps, factual, last_factual, None, None, None, None, None)
            else:
                dev_memory.push(state, action, next_state, reward,
                                done, isweight, None, n_steps, factual, last_factual, None, None, None, None, None)
            traj_set.push(state, action, next_state, reward, done, isweight, None,
                          n_steps, factual, last_factual, None, None, None, None, None)
            n_steps += 1
    memory.flatten() # prepare flatten data
    dev_memory.flatten()
    memory.update_u() # prepare u_{0:t}
    dev_memory.update_u()
    t1 = time()
    print("Generating {} trajectories took {} minutes".format(
        config.sample_num_traj, (t1-t0)/60))
    return memory, dev_memory, traj_set, scores

def train_model(memory, dev_memory, config, loss_mode):
    mdpnet = MDPnet(config)

    best_train_loss = 100
    lr = config.lr
    train_loss_list = []
    dev_loss_list = []
    for i_episode in range(config.train_num_episodes):
        train_loss = 0
        dev_loss = 0
        optimizer = optim.Adam(mdpnet.parameters(), lr=lr)
        for i_batch in range(config.train_num_batches):
            train_loss_batch = mdpmodel_train(memory, mdpnet, optimizer,
                                              loss_mode, config)
            train_loss = ((train_loss * i_batch + train_loss_batch)
                          / (i_batch + 1))
            if config.dev_num_traj > 0:
                dev_loss_batch = mdpmodel_test(dev_memory, mdpnet, 0,
                                               config)
                dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        if (i_episode+1) % config.print_per_epi == 0:
            print('Episode {}: train loss {:.3e}, dev loss {:.3e}'.format(
                i_episode+1, train_loss, dev_loss))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        else:
            lr *= config.lr_decay
        train_loss_list += [train_loss]
        dev_loss_list += [dev_loss]

    return train_loss_list, dev_loss_list, mdpnet


def rollout_batch(env, config, init_states, mdpnet, num_rollout,
                  init_done=None, init_actions=None):
    ori_batch_size = init_states.size()[0]
    batch_size = init_states.size()[0] * num_rollout
    init_states = init_states.repeat(num_rollout, 1)
    if init_actions is not None:
        init_actions = init_actions.repeat(num_rollout, 1)
    if init_done is not None:
        init_done = init_done.repeat(num_rollout)
    states = init_states
    if init_done is None:
        done = ByteTensor(batch_size)
        done.fill_(0)
    else:
        done = init_done
    if init_actions is None:
        actions = []
        for i_actions in range(batch_size):
            actions.append(int(env.policy(states[i_actions],
                                          eps=0)))
    else:
        actions = init_actions
    n_steps = 0
    t_reward = torch.zeros(batch_size)
    done = False
    while not done:
        if n_steps > 0:
            actions = []
            for i_actions in range(batch_size):
                actions.append(int(env.policy(states[i_actions], eps=0)))
        states = Variable(Tensor(states))
        states_diff, reward, _ = mdpnet.forward(states)
        states_diff = states_diff.data
        actions = LongTensor(actions)
        reward = reward.data.gather(1, actions.view(-1, 1)).squeeze()
        expanded_actions = actions.view(-1, 1).unsqueeze(2)
        expanded_actions = expanded_actions.expand(-1, -1, config.state_dim)
        states_diff = states_diff.gather(1, expanded_actions).squeeze()
        # state_diff = state_diff.view(-1, config.state_dim)
        next_states = states_diff + states.data
        states = next_states
        t_reward = t_reward + config.gamma**n_steps * reward
        # done_this_step = is_done.forward(Variable(states)).data[:,0] > 0
        done = n_steps == config.max_length - 1
        n_steps += 1
    value = t_reward.numpy()
    value = np.reshape(value, [num_rollout, ori_batch_size])
    return np.mean(value, 0)


def compute_values(env, traj_set, model, config, model_type='MDP'):
    num_samples = len(traj_set)
    traj_len = np.zeros(num_samples, 'int')
    state_tensor = FloatTensor(num_samples, config.max_length, config.state_dim).zero_()
    action_tensor = LongTensor(num_samples, config.max_length, 1).zero_()
    done_tensor = ByteTensor(num_samples, config.max_length).fill_(1)
    V_value = np.zeros((num_samples, config.max_length))
    Q_value = np.zeros((num_samples, config.max_length))

    for i_traj in range(num_samples):
        traj_len[i_traj] = len(traj_set.trajectories[i_traj])
        state_tensor[i_traj,0:traj_len[i_traj],:] = torch.cat([ t.state for t in traj_set.trajectories[i_traj] ])
        action_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.action for t in traj_set.trajectories[i_traj]])
        done_tensor[i_traj, 0:traj_len[i_traj] ].fill_(0)
        if traj_len[i_traj] < config.max_length:
            done_tensor[i_traj, traj_len[i_traj] :].fill_(1)

    # Cut off unnecessary computation, if at a time step t all IS weights are zero
    nonzero_is = np.zeros(config.max_length,'int')
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            if w > 0:
                nonzero_is[t.time] += 1
            w *= t.isweight[0]

    for i_step in range(config.max_length):
        # if nonzero_is[i_step] == 0:
        #     break
        if model_type == 'MDP':
            V_value[:, i_step] = rollout_batch(env=env, init_states=state_tensor[:, i_step, :], mdpnet=model,
                                               num_rollout=config.eval_num_rollout, config=config,
                                               init_done=done_tensor[:, i_step])
            Q_value[:, i_step] = rollout_batch(env=env, init_states=state_tensor[:, i_step, :], mdpnet=model,
                                               num_rollout=config.eval_num_rollout, config=config,
                                               init_done=done_tensor[:, i_step],
                                               init_actions=action_tensor[:, i_step, :])
        elif model_type == 'IS':
            pass
    return V_value, Q_value


def doubly_robust(traj_set, V_value, Q_value, config, wis=False):
    num_samples = len(traj_set)
    weights = np.zeros((num_samples,config.max_length))
    weights_sum = np.zeros(config.max_length)

    for i_traj in range(num_samples):
        for n in range(config.max_length):
            if n >= len(traj_set.trajectories[i_traj]):
                weights[i_traj:,n] = weights[i_traj,n-1]
                break
            if n == 0:
                weights[i_traj, n] = 1.0
            else:
                weights[i_traj,n] = weights[i_traj,n-1]*traj_set.trajectories[i_traj][n].isweight[0].item()

    if wis:
        for n in range(config.max_length):
            weights_sum[n] = np.sum(weights[:,n])
            if weights_sum[n] != 0:
                weights[:,n] = (weights[:,n]*num_samples)/weights_sum[n]

    value = np.zeros(num_samples)
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            #print(t.reward[0].item(), t.time, weights[i_traj,t.time], Q_value[i_traj,t.time], V_value[i_traj,t.time])
            value[i_traj] += weights[i_traj,t.time]*(t.reward[0].item() - Q_value[i_traj,t.time]) + w*V_value[i_traj,t.time]
            w = weights[i_traj,t.time]
            if w == 0:
                break
    return value


def importance_sampling(traj_set, wis=False):
    num_samples = len(traj_set)
    value = np.zeros(num_samples)
    weights = np.zeros(num_samples)
    for i_traj in range(num_samples):
        l = len(traj_set.trajectories[i_traj])
        tmp = 1
        for n in range(l):
            tmp *= traj_set.trajectories[i_traj][n].isweight[0].item()
        weights[i_traj] = tmp

    if wis:
        weights = (weights*num_samples)/np.sum(weights)

    for i_traj in range(num_samples):
        l = len(traj_set.trajectories[i_traj])
        value[i_traj] = l*weights[i_traj]
    return value


if __name__ == "__main__":

    estm_list = []
    estm_bsl_list = []
    wdr_list = []
    wdr_bsl_list = []
    ips_list = []
    pdis_list = []
    wpdis_list = []

    empirical_scores_list = []
    config = hiv_config
    if args.train_num_traj:
        config.train_num_traj = args.train_num_traj
        config.dev_num_traj = args.dev_num_traj
        config.sample_num_traj = args.train_num_traj + args.dev_num_traj
    else:
        config.sample_num_traj = config.train_num_traj + config.dev_num_traj
    if config.train_batch_size > config.train_num_traj:
        config.train_batch_size = config.train_num_traj
    """ Load hiv environment - the environment comes with a policy which can be
    made eps greedy. """
    with open('hiv_domain/hiv_simulator/hiv_preset_hidden_params', 'rb') as f:
        preset_hidden_params = pickle.load(f, encoding='latin1')
    env = FittedQIteration(perturb_rate=0.05,
                           preset_params=preset_hidden_params[config.ins],
                           gamma=config.gamma,
                           ins=config.ins,
                           episode_length=config.max_length)
    env.tree = joblib.load('hiv_domain/extra_tree_gamma_ins20.pkl')

    eval_env = FittedQIteration(perturb_rate=0.05,
                           preset_params=preset_hidden_params[config.ins],
                           gamma=config.gamma,
                           ins=config.ins,
                           episode_length=config.max_length)
    eval_env.tree= joblib.load('hiv_domain/extra_tree_gamma_ins20.pkl')

    if config.fix_data:
        memory, dev_memory, traj_set, scores = generate_data(env, config, eval_env)

    if args.N:
        config.N = args.N
    for i in range(config.N):
        print("exp {}".format(i+1))
        if not config.fix_data:
            memory, dev_memory, traj_set, scores = generate_data(env, config, eval_env)

        print('Learn our mdp model')
        train_loss_list, dev_loss_list, mdpnet = (
            train_model(memory, dev_memory, config, 1))
        print('Learn the baseline mdp model')
        train_loss_list, dev_loss_list, mdpnet_unweight = (
            train_model(memory, dev_memory,  config, 0))

        print('Evaluate models using evaluation policy on the same initial states')
        mdpnet.eval()
        mdpnet_unweight.eval()

        init_states = []
        for i_episode in range(config.eval_pib_num_rollout):
            env.task.reset(perturb_params=True,
                           **preset_hidden_params[config.ins])
            init_states.append(preprocess_state(env.task.observe(),
                                                config.state_dim))
        init_states = torch.cat(init_states)
        estm = rollout_batch(eval_env, config, init_states, mdpnet,
                             config.eval_num_rollout)
        estm_bsl = rollout_batch(eval_env, config, init_states, mdpnet_unweight,
                                 config.eval_num_rollout)

        # V,Q = compute_values(eval_env, traj_set, mdpnet, config, model_type='MDP')
        # wdr = doubly_robust(traj_set, V, Q, config, wis=True)
        #
        # V, Q = compute_values(eval_env, traj_set, mdpnet_unweight, config, model_type='MDP')
        # wdr_bsl = doubly_robust(traj_set, V, Q, config, wis=True)

        V, Q = compute_values(eval_env, traj_set, None, config, model_type='IS')
        ips = importance_sampling(traj_set)
        pdis = doubly_robust(traj_set, V, Q, config, wis=False)
        wpdis = doubly_robust(traj_set, V, Q, config, wis=True)

        with open("hiv_domain/standardization_data.pkl", "rb") as fobj:
            standardization_data = pickle.load(fobj)
            reward_mean = standardization_data[config.ins]["reward_mean"]
            reward_std = standardization_data[config.ins]["reward_std"]

        estm_rescaled = np.array(estm)
        estm_rescaled = estm_rescaled * reward_std + reward_mean * (1-config.gamma**config.max_length)/(1-config.gamma)
        estm_bsl_rescaled = np.array(estm_bsl)
        estm_bsl_rescaled = estm_bsl_rescaled * reward_std + reward_mean * (1-config.gamma**config.max_length)/(1-config.gamma)

        # wdr_rescaled = np.array(wdr)
        # wdr_rescaled = wdr_rescaled * reward_std + reward_mean * (1 - config.gamma ** config.max_length) / (
        # 1 - config.gamma)
        # wdr_bsl_rescaled = np.array(wdr_bsl)
        # wdr_bsl_rescaled = wdr_bsl_rescaled * reward_std + reward_mean * (1 - config.gamma ** config.max_length) / (
        # 1 - config.gamma)

        ips_rescaled = np.array(ips)
        ips_rescaled = ips_rescaled * reward_std + reward_mean * (1 - config.gamma ** config.max_length) / (
            1 - config.gamma)
        pdis_rescaled = np.array(pdis)
        pdis_rescaled = pdis_rescaled * reward_std + reward_mean * (1 - config.gamma ** config.max_length) / (
            1 - config.gamma)
        wpdis_rescaled = np.array(wpdis)
        wpdis_rescaled = wpdis_rescaled * reward_std + reward_mean * (1 - config.gamma ** config.max_length) / (
            1 - config.gamma)


        print("RepBM MDP model", np.mean(estm_rescaled))
        print("MDP model", np.mean(estm_bsl_rescaled))
        # print(np.mean(wdr_rescaled))
        # print(np.mean(wdr_bsl_rescaled))
        print("IS", np.mean(ips_rescaled))
        print("PSIS", np.mean(pdis_rescaled))
        print("WPSIS", np.mean(wpdis_rescaled))

        estm_list += [np.mean(estm_rescaled)]
        estm_bsl_list += [np.mean(estm_bsl_rescaled)]
        # wdr_list += [np.mean(wdr_rescaled)]
        # wdr_bsl_list += [np.mean(wdr_bsl_rescaled)]
        ips_list += [np.mean(ips_rescaled)]
        pdis_list += [np.mean(pdis_rescaled)]
        wpdis_list += [np.mean(wpdis_rescaled)]

    if args.train_num_traj:
        fname = "results/temp_data_hiv_N" + str(args.train_num_traj) + "_" + str(args.file_num) + ".npz"
    else:
        fname = "results/data_hiv.npz"

    np.savez(fname,
             estm_list=estm_list,
             estm_bsl_list=estm_bsl_list,
             ips_list=ips_list,
             pdis_list=pdis_list,
             wpdis_list=wpdis_list
             )