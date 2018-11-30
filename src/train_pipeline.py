import time
from src.models import QtNet, MDPnet, TerminalClassifier
from src.memory import MRDRTransition, TrajectorySet, SampleSet
from src.utils import *
from collections import deque
import torch.optim as optim

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Implementation of more robust doubly robust


def mrdr_loss(q_values, onehot_rewards, weights, pib, pie):
    batch_size = q_values.size()[0]
    action_size = q_values.size()[1]
    Omega = FloatTensor(batch_size, action_size, action_size).zero_()
    for i in range(action_size):
        Omega[:,i,i] = 1/pib.detach()[:,i]
    Omega = Omega - torch.ones(batch_size, action_size, action_size)

    q = pie*q_values - onehot_rewards
    q = q.unsqueeze(2) # batch_size x action_size x 1

    error = torch.bmm(Omega,q)
    error = torch.bmm(torch.transpose(q,1,2),error)
    error = error.squeeze()
    loss = error * weights
    loss = loss.mean(0)
    return loss


def mrdr_train(samples, qnet, optimizer, loss_mode, config, permutation, i_batch, wis_reward=False):
    qnet.train()
    start = (i_batch*config.mrdr_batch_size)%len(permutation)
    if start + config.mrdr_batch_size >= len(permutation):
        start = 0
    transitions = [samples[i] for i in permutation[start:start+config.mrdr_batch_size]]
    num_samples = config.mrdr_batch_size
    batch = MRDRTransition(*zip(*transitions))

    states = torch.cat(batch.state)
    times = torch.cat([Tensor([[t.time]]) for t in transitions])/config.max_length

    actions = torch.cat(batch.action) # batch_size x 1
    pib = torch.cat(batch.pib) # batch_size x action_size
    if loss_mode == 's':
        pie = torch.cat(batch.soft_pie)
    else:
        pie = torch.cat(batch.pie)
    # R_{0:t}
    if wis_reward:
        acc_rewards = torch.cat(batch.wacc_reward) / config.max_length  # batch_size x,
    else:
        acc_rewards = torch.cat(batch.acc_reward)/config.max_length # batch_size x,
    onehot_rewards = FloatTensor(num_samples,config.action_size).zero_() # batch_size x action_size
    onehot_rewards.scatter_(1, actions, acc_rewards.unsqueeze(1))

    # w_{0:t-1}^2 * w_t
    w_0tot = torch.cat([t.acc_soft_isweight for t in transitions])
    w_t = torch.cat([t.soft_isweight for t in transitions])
    weights = w_0tot**2 / w_t # batch_size x,

    # Q(s_t, .)
    q_values = qnet(states.type(Tensor), times) # batch x action_size
    if loss_mode == 's': # stochastic: add some noise into pie to create a biased but lower variance target
        loss = mrdr_loss(q_values, onehot_rewards, weights, pib, pie)
    elif loss_mode == 'd': # deterministic: original, deterministic pie
        pib_a = pib.gather(1,actions).squeeze()
        weights = (weights*(1-pib_a)/(pib_a**2))
        qa_values = q_values.gather(1,actions).squeeze()
        loss = weighted_mse_loss(qa_values,acc_rewards,weights)
    elif loss_mode == 'b': # behavior: use pib as pie in the MRDR objective function
        qa_values = q_values.gather(1, actions).squeeze()
        loss = torch.nn.MSELoss()(qa_values, acc_rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def mrdr_test(samples, qnet, loss_mode, config, permutation, i_batch, wis_reward=False):
    qnet.train()
    #transitions = random.sample(samples, config.mrdr_test_batch_size)
    start = (i_batch*config.mrdr_test_batch_size)%len(permutation)
    if start + config.mrdr_test_batch_size >= len(permutation):
        start = 0
    transitions = [samples[i] for i in permutation[start:start+config.mrdr_test_batch_size]]
    num_samples = config.mrdr_test_batch_size
    batch = MRDRTransition(*zip(*transitions))

    states = torch.cat(batch.state)
    times = torch.cat([Tensor([[t.time]]) for t in transitions])/config.max_length
    actions = torch.cat(batch.action) # batch_size x 1
    pib = torch.cat(batch.pib) # batch_size x action_size
    if loss_mode == 's':
        pie = torch.cat(batch.soft_pie)
    else:
        pie = torch.cat(batch.pie)
    # R_{0:t}
    if wis_reward:
        acc_rewards = torch.cat(batch.wacc_reward) / config.max_length  # batch_size x,
    else:
        acc_rewards = torch.cat(batch.acc_reward)/config.max_length # batch_size x,
    onehot_rewards = FloatTensor(num_samples,config.action_size).zero_() # batch_size x action_size
    onehot_rewards.scatter_(1, actions, acc_rewards.unsqueeze(1))

    # w_{0:t-1}^2 * w_t
    w_0tot = torch.cat([t.acc_soft_isweight for t in transitions])
    w_t = torch.cat([t.soft_isweight for t in transitions])
    weights = w_0tot**2 / w_t # batch_size x,

    # Q(s_t, .)
    q_values = qnet(states.type(Tensor), times) # batch x action_size
    if loss_mode == 's':
        loss = mrdr_loss(q_values, onehot_rewards, weights, pib, pie)
    elif loss_mode == 'd':
        pib_a = pib.gather(1, actions).squeeze()
        weights = weights*(1-pib_a)/(pib_a**2)
        qa_values = q_values.gather(1,actions).squeeze()
        loss = weighted_mse_loss(qa_values,acc_rewards,weights)
    elif loss_mode == 'b':
        qa_values = q_values.gather(1, actions).squeeze()
        loss = torch.nn.MSELoss()(qa_values, acc_rewards)
    return loss.item()


# Implementation of RepBM and baseline MDP model


def mdpmodel_train(memory, mdpnet, optimizer, loss_mode, config):
    mdpnet.train()
    # Adapted from pytorch tutorial example code
    if config.train_batch_size > len(memory):
        return
    transitions = memory.sample(config.train_batch_size)
    time = transitions[0].time
    ratio = 1/memory.u[time] if memory.u[time] != 0 else 0

    next_states = torch.cat([t.next_state for t in transitions])
    states = torch.cat([t.state for t in transitions]) # batch_size x state_dim
    states_diff = (next_states-states)  # batch_size x state_dim
    actions = torch.cat([t.action for t in transitions])  # batch_size x 1
    reward = torch.cat([t.reward for t in transitions])  # batch_size or batch_size x 1
    reward = reward.squeeze() # batch_size

    factual = torch.cat([t.factual for t in transitions])
    weights = FloatTensor(config.train_batch_size)
    weights.fill_(1.0)
    weights = weights + factual*ratio
    weights_2 = factual*ratio

    # Compute T(s_t, a) - the model computes T(s_t), then we select the
    # columns of actions taken
    expanded_actions = actions.unsqueeze(2) # batch_size x 1 x 1
    expanded_actions = expanded_actions.expand(-1,-1,config.state_dim) # batch_size x 1 x state_dim
    predict_state_diff, predict_reward_value, rep = mdpnet(states.type(Tensor))
    # predict_state_diff = batch_size x 2 x state_dim
    predict_state_diff = predict_state_diff.gather(1, expanded_actions).squeeze() # batch_size x state_dim
    predict_reward_value = predict_reward_value.gather(1, actions).squeeze()

    sum_factual = sum([t.factual[0] for t in transitions])
    sum_control = sum([t.last_factual[0] for t in transitions])
    if sum_factual > 0 and sum_control > 0:
        factual_index = LongTensor([i for i, t in enumerate(transitions) if t.factual[0] == 1])
        last_factual_index = LongTensor([i for i, t in enumerate(transitions) if t.last_factual[0] == 1])
        factual_rep = rep.index_select(0,factual_index)
        control_rep = rep.index_select(0,last_factual_index)
        loss_rep = mmd_lin(factual_rep, control_rep)
    else:
        loss_rep = 0

    # Compute loss
    if loss_mode == 0: # MSE_mu
        loss = torch.nn.MSELoss()(predict_state_diff, states_diff) \
               + torch.nn.MSELoss()(predict_reward_value, reward)
    elif loss_mode == 1: # MSE_pi + MSE_mu (objective in RepBM paper)
        loss = weighted_mse_loss(predict_state_diff, states_diff, weights) \
               + weighted_mse_loss(predict_reward_value, reward, weights) \
               + config.alpha_rep*loss_rep #+ config.alpha_bce*torch.nn.BCELoss(weights)(predict_soft_done,done)
    elif loss_mode == 2: # MSE_pi: only MSE_pi, as an ablation study
        loss = weighted_mse_loss(predict_state_diff, states_diff, weights_2) \
               + weighted_mse_loss(predict_reward_value, reward, weights_2)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in mdpnet.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def mdpmodel_test(memory, mdpnet, loss_mode, config):
    mdpnet.eval()
    # Adapted from pytorch tutorial example code
    if config.test_batch_size > len(memory):
        return
    transitions = memory.sample(config.test_batch_size)
    time = transitions[0].time
    ratio = 1 / memory.u[time] if memory.u[time] != 0 else 0

    # Compute a mask of non-final states and concatenate the batch elements
    next_states = torch.cat([t.next_state for t in transitions])
    states = torch.cat([t.state for t in transitions])
    states_diff = (next_states - states)
    actions = (torch.cat([t.action for t in transitions]))
    reward = torch.cat([t.reward for t in transitions])
    reward = reward.squeeze()

    sum_factual = sum([t.factual[0] for t in transitions])
    sum_control = sum([t.last_factual[0] for t in transitions])

    factual = torch.cat([t.factual for t in transitions])
    weights = FloatTensor(config.test_batch_size)
    weights.fill_(1.0)
    weights = weights + factual * ratio
    weights_2 = factual * ratio

    # Compute T(s_t, a) - the model computes T(s_t), then we select the
    # columns of actions taken
    expanded_actions = actions.unsqueeze(2)
    expanded_actions = expanded_actions.expand(-1, -1, config.state_dim)
    predict_state_diff, predict_reward_value, rep = mdpnet(states.type(Tensor))
    predict_state_diff = predict_state_diff.gather(1, expanded_actions).squeeze()
    predict_reward_value = predict_reward_value.gather(1, actions).squeeze()

    if sum_factual > 0 and sum_control > 0:
        factual_index = LongTensor([i for i, t in enumerate(transitions) if t.factual[0] == 1])
        last_factual_index = LongTensor([i for i, t in enumerate(transitions) if t.last_factual[0] == 1])
        factual_rep = rep.index_select(0, factual_index)
        control_rep = rep.index_select(0, last_factual_index)
        loss_rep = mmd_lin(factual_rep, control_rep)
    else:
        loss_rep = 0

    # Compute loss
    loss_mode = 0
    if loss_mode == 0:
        loss = torch.nn.MSELoss()(predict_state_diff, states_diff) \
               + torch.nn.MSELoss()(predict_reward_value, reward)
    elif loss_mode == 1:
        loss = weighted_mse_loss(predict_state_diff, states_diff, weights) \
               + weighted_mse_loss(predict_reward_value, reward, weights) \
               + config.alpha_rep*loss_rep
    elif loss_mode == 2:
        loss = weighted_mse_loss(predict_state_diff, states_diff, weights_2) \
               + weighted_mse_loss(predict_reward_value, reward, weights_2)
    return loss.item()


def terminal_classifier_train(memory, model, optimizer, batch_size):
    model.train()
    transitions_origin = memory.sample(batch_size)
    transitions_terminal = memory.sample_terminal(batch_size)
    transitions = transitions_origin + transitions_terminal
    num_samples = len(transitions)

    states = torch.cat([t.state for t in transitions])
    next_states = torch.cat([t.next_state for t in transitions])
    all_states = torch.cat((next_states,states),0)
    done = torch.cat([FloatTensor([[t.done and t.time < 199]]) for t in transitions])
    all_done = torch.cat((done, torch.zeros(num_samples,1)),0)
    predict_soft_done = model(all_states.type(Tensor))

    # Optimize the model
    loss = torch.nn.BCEWithLogitsLoss()(predict_soft_done, all_done)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = ((predict_soft_done.detach()[:,0]>0).float()==all_done[:,0]).sum().item()/(all_done.size()[0])
    return loss.item(), acc


def terminal_classifier_test(memory, model, batch_size):
    model.eval()
    transitions_origin = memory.sample(batch_size)
    transitions_terminal = memory.sample_terminal(batch_size)
    transitions = transitions_origin + transitions_terminal
    num_samples = len(transitions)

    states = torch.cat([t.state for t in transitions])
    next_states = torch.cat([t.next_state for t in transitions])
    all_states = torch.cat((next_states,states), 0)
    done = torch.cat([FloatTensor([[t.done and t.time < 199]]) for t in transitions])
    all_done = torch.cat((done, torch.zeros(num_samples, 1)), 0)
    predict_soft_done = model(all_states.type(Tensor))

    # Optimize the model
    loss = torch.nn.BCEWithLogitsLoss()(predict_soft_done, all_done)
    acc = ((predict_soft_done.detach()[:, 0] > 0).float() == all_done[:, 0]).sum().item() / (all_done.size()[0])
    return loss.item(), acc


def rollout_batch(init_states, mdpnet, is_done, num_rollout, policy_qnet, epsilon, action_size, maxlength, config, init_done=None, init_actions=None):
    ori_batch_size = init_states.size()[0]
    state_dim = init_states.size()[1]
    batch_size = init_states.size()[0]*num_rollout
    init_states = init_states.repeat(num_rollout,1)
    if init_actions is not None:
        init_actions = init_actions.repeat(num_rollout,1)
    if init_done is not None:
        init_done = init_done.repeat(num_rollout)
    states = init_states
    if init_done is None:
        done = ByteTensor(batch_size)
        done.fill_(0)
    else:
        done = init_done
    if init_actions is None:
        actions = epsilon_greedy_action_batch(states, policy_qnet, epsilon, action_size)
    else:
        actions = init_actions
    n_steps = 0
    t_reward = torch.zeros(batch_size)
    while not (sum(done.long() == batch_size) or n_steps >= maxlength):
        if n_steps > 0:
            actions = epsilon_greedy_action_batch(states, policy_qnet, epsilon, action_size)
        states_re = Tensor(np.float32(config.rescale)*states)
        states_diff, reward, _ = mdpnet.forward(states_re.type(Tensor))
        states_diff = states_diff.detach()
        reward = reward.detach().gather(1, actions).squeeze()
        expanded_actions = actions.unsqueeze(2)
        expanded_actions = expanded_actions.expand(-1, -1, state_dim)
        states_diff = states_diff.gather(1, expanded_actions).squeeze()
        states_diff = states_diff.view(-1, config.state_dim)
        states_diff = 1 / np.float32(config.rescale) * states_diff
        next_states = states_diff + states
        states = next_states
        t_reward = t_reward + (1 - done).float() * reward
        done = done | (is_done.forward(states.type(Tensor)).detach()[:, 0] > 0)
        n_steps += 1
    value = t_reward.numpy()
    value = np.reshape(value,[num_rollout,ori_batch_size])
    return np.mean(value,0)


def compute_values(traj_set, model, is_done, policy_qnet, config, model_type='MDP', soften=False):
    num_samples = len(traj_set)
    traj_len = np.zeros(num_samples, 'int')
    state_tensor = FloatTensor(num_samples, config.max_length, config.state_dim).zero_()
    action_tensor = LongTensor(num_samples, config.max_length, 1).zero_()
    pie_tensor = FloatTensor(num_samples, config.max_length, config.action_size).zero_()
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
        if soften:
            pie_tensor[i_traj, 0:traj_len[i_traj], :] = torch.cat([t.soft_pie for t in traj_set.trajectories[i_traj]])

    # Cut off unnecessary computation: if a time step t
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
            if soften:
                V_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet, epsilon=config.soften_epsilon, action_size=config.action_size,
                                                   maxlength=config.max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step])
                Q_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet,
                                                   epsilon=config.soften_epsilon, action_size=config.action_size,
                                                   maxlength=config.max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step],
                                                   init_actions=action_tensor[:, i_step, :])
            else:
                V_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet, epsilon=0, action_size=config.action_size,
                                                   maxlength=config.max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step])
                Q_value[:, i_step] = rollout_batch(state_tensor[:, i_step, :], model, is_done, config.eval_num_rollout,
                                                   policy_qnet,
                                                   epsilon=0, action_size=config.action_size,
                                                   maxlength=config.max_length - i_step, config=config,
                                                   init_done=done_tensor[:, i_step],
                                                   init_actions=action_tensor[:, i_step, :])
        elif model_type == 'Q':
            times = Tensor(num_samples,1).fill_(i_step)/config.max_length
            q_values = model.forward(state_tensor[:, i_step, :], times).detach()*config.max_length
            pie_actions = epsilon_greedy_action_batch(state_tensor[:, i_step, :], policy_qnet, 0, config.action_size)
            if soften:
                zeros_actions = LongTensor(num_samples, 1).zero_()
                ones_actions = 1-zeros_actions
                V_value[:, i_step] = pie_tensor[:, i_step, 0] * q_values.gather(1, zeros_actions).squeeze() \
                                     + pie_tensor[:, i_step, 1] * q_values.gather(1, ones_actions).squeeze()
            else:
                V_value[:, i_step] = q_values.gather(1, pie_actions).squeeze()
            Q_value[:, i_step] = q_values.gather(1,action_tensor[:, i_step, :]).squeeze()
        elif model_type == 'IS':
            pass
    return V_value, Q_value


def doubly_robust(traj_set, V_value, Q_value, config, wis=False, soften=False):
    num_samples = len(traj_set)
    weights = np.zeros((num_samples,config.max_length))
    weights_sum = np.zeros(config.max_length)

    for i_traj in range(num_samples):
        for n in range(config.max_length):
            if n >= len(traj_set.trajectories[i_traj]):
                weights[i_traj:,n] = weights[i_traj,n-1]
                break
            if soften:
                weights[i_traj,n] = traj_set.trajectories[i_traj][n].acc_soft_isweight[0].item()
            else:
                weights[i_traj,n] = traj_set.trajectories[i_traj][n].acc_isweight[0].item()

    if wis:
        for n in range(config.max_length):
            weights_sum[n] = np.sum(weights[:,n])
            if weights_sum[n] != 0:
                weights[:,n] = (weights[:,n]*num_samples)/weights_sum[n]

    value = np.zeros(num_samples)
    for i_traj in range(num_samples):
        w = 1
        for t in traj_set.trajectories[i_traj]:
            value[i_traj] += weights[i_traj,t.time]*(t.reward[0].item() - Q_value[i_traj,t.time]) + w*V_value[i_traj,t.time]
            w = weights[i_traj,t.time]
            if w == 0:
                break
    return value


def importance_sampling(traj_set, wis=False, soften=False):
    num_samples = len(traj_set)
    value = np.zeros(num_samples)
    weights = np.zeros(num_samples)
    for i_traj in range(num_samples):
        l = len(traj_set.trajectories[i_traj])
        if soften:
            weights[i_traj] = traj_set.trajectories[i_traj][l - 1].acc_soft_isweight[0].item()
        else:
            weights[i_traj]=traj_set.trajectories[i_traj][l-1].acc_isweight[0].item()
    if wis:
        weights = (weights*num_samples)/np.sum(weights)

    for i_traj in range(num_samples):
        l = len(traj_set.trajectories[i_traj])
        value[i_traj] = l*weights[i_traj]
    return value


def mrdr_preprocess(traj_set, config):
    transitions = []
    weights = np.zeros((len(traj_set),config.max_length,config.max_length))
    weights_sum = np.zeros((config.max_length,config.max_length))
    weights_num = np.zeros((config.max_length,config.max_length))
    for i in range(len(traj_set)):
        for n1 in range(config.max_length):
            if n1 >= len(traj_set.trajectories[i]):
                weights[i,n1:,:] = 0
                break
            for n2 in range(n1, config.max_length):
                if n2 >= len(traj_set.trajectories[i]):
                    weights[i,n1,n2:] = weights[i,n1,n2-1]
                    weights_sum[n1,n2:] += weights[i,n1,n2-1]
                    weights_num[n1,n2:] += 1
                    break
                weights[i,n1,n2] = traj_set.trajectories[i][n2].acc_soft_isweight\
                                   /traj_set.trajectories[i][n1].acc_soft_isweight
                weights_sum[n1, n2] += weights[i, n1, n2]
                weights_num[n1, n2] += 1

    for i in range(len(traj_set)):
        traj = traj_set.trajectories[i]
        for n1 in range(len(traj)):
            R = 0
            WR = 0
            for n2 in range(n1, len(traj)):
                R += traj[n2].reward*weights[i,n1,n2]
                WR += traj[n2].reward*weights[i,n1,n2]*weights_num[n1,n2]/weights_sum[n1,n2]
                #print(weights[i,n1,n2], weights[i,n1,n2]*weights_num[n1,n2]/weights_sum[n1,n2])
                #print(n1, n2, weights_num[n1,n2], weights_sum[n1,n2])
            transitions.append(MRDRTransition(*traj[n1], R, WR))
            #print(n1, R[0], WR[0])
    return transitions


def train_pipeline(env, config, eval_qnet, seedvec = None):
    memory = SampleSet(config) # same the tuples for model training
    dev_memory = SampleSet(config)
    traj_set = TrajectorySet(config) # save the trajectory for doubly robust evaluation
    scores = deque()
    mdpnet_msepi = MDPnet(config)
    mdpnet = MDPnet(config)
    mdpnet_unweight = MDPnet(config)
    mrdr_q = QtNet(config.state_dim,config.mrdr_hidden_dims,config.action_size)
    mrdrv2_q = QtNet(config.state_dim, config.mrdr_hidden_dims, config.action_size)
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
            q_values = eval_qnet.forward(state.type(Tensor)).detach()
            action = epsilon_greedy_action(state, eval_qnet, config.behavior_epsilon, config.action_size, q_values)
            p_pib = epsilon_greedy_action_prob(state, eval_qnet, config.behavior_epsilon,
                                               config.action_size, q_values)
            soft_pie = epsilon_greedy_action_prob(state, eval_qnet, config.soften_epsilon,
                                               config.action_size, q_values)
            p_pie = epsilon_greedy_action_prob(state, eval_qnet, 0, config.action_size, q_values)

            isweight = p_pie[:, action.item()] / p_pib[:,action.item()]
            acc_isweight = acc_isweight * (p_pie[:, action.item()] / p_pib[:, action.item()])
            soft_isweight = (soft_pie[:, action.item()] / p_pib[:, action.item()])
            acc_soft_isweight = acc_soft_isweight * (soft_pie[:, action.item()] / p_pib[:, action.item()])

            last_factual = factual * (1 - p_pie[:, action.item()]) # 1{a_{0:t-1}==\pie, a_t != \pie}
            factual = factual * p_pie[:, action.item()] # 1{a_{0:t}==\pie}

            next_state, reward, done, _ = env.step(action.item())
            reward = Tensor([reward])
            next_state = preprocess_state(next_state, config.state_dim)
            next_state_re = np.float32(config.rescale)*next_state
            state_re = np.float32(config.rescale)*state
            if i_episode < config.train_num_traj:
                memory.push(state_re, action, next_state_re, reward, done, isweight, acc_isweight, n_steps, factual,
                            last_factual, acc_soft_isweight, soft_isweight, soft_pie, p_pie, p_pib)
            else:
                dev_memory.push(state_re, action, next_state_re, reward, done, isweight, acc_isweight, n_steps, factual,
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
    #print(len(memory.terminal))
    time_now = time.time()
    time_sampling = time_now-time_pre
    time_pre = time_now

    mrdr_samples = mrdr_preprocess(traj_set, config)
    total_num = len(mrdr_samples)
    mrdr_train_samples = mrdr_samples[: int(total_num*0.9)]
    mrdr_dev_samples = mrdr_samples[int(total_num*0.9) :]
    time_now = time.time()
    time_premrdr = time_now - time_pre
    time_pre = time_now

    print('Learn MRDR Q functions')
    best_train_loss = 10000
    lr = config.mrdr_lr
    optimizer = optim.Adam(mrdr_q.parameters(), lr=lr)
    for i_episode in range(config.mrdr_num_episodes):
        train_loss = 0
        dev_loss = 0
        train_permutation = np.random.permutation(len(mrdr_train_samples))
        dev_permutation = np.random.permutation(len(mrdr_dev_samples))
        for i_batch in range(config.mrdr_num_batches):
            train_loss_batch = mrdr_train(mrdr_train_samples, mrdr_q, optimizer,
                                          'd', config, train_permutation, i_batch, wis_reward=False)
            dev_loss_batch = mrdr_test(mrdr_dev_samples, mrdr_q,
                                       'd', config, dev_permutation, i_batch, wis_reward=False)
            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        if (i_episode + 1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}, lr={}'
                  .format(i_episode + 1, train_loss, dev_loss, lr))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        else:
            lr *= 0.9
            learning_rate_update(optimizer, lr)
    time_now = time.time()
    time_mrdr = time_now - time_pre
    time_pre = time_now

    print('Learn MRDR-WIS Q functions')
    best_train_loss = 10000
    lr = config.mrdr_lr
    optimizer = optim.Adam(mrdrv2_q.parameters(), lr=lr)
    for i_episode in range(config.mrdr_num_episodes):
        train_loss = 0
        dev_loss = 0
        train_permutation = np.random.permutation(len(mrdr_train_samples))
        dev_permutation = np.random.permutation(len(mrdr_dev_samples))
        for i_batch in range(config.mrdr_num_batches):
            train_loss_batch = mrdr_train(mrdr_train_samples, mrdrv2_q, optimizer,
                                          'd', config, train_permutation, i_batch, wis_reward=True)
            dev_loss_batch = mrdr_test(mrdr_dev_samples, mrdrv2_q,
                                       'd', config, dev_permutation, i_batch, wis_reward=True)
            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        if (i_episode + 1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}, lr={}'
                  .format(i_episode + 1, train_loss, dev_loss, lr))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        else:
            lr *= 0.9
            learning_rate_update(optimizer, lr)
    time_now = time.time()
    time_mrdr += time_now - time_pre
    time_pre = time_now

    print('Learn mse_pi mdp model')
    best_train_loss = 100
    lr = config.lr
    for i_episode in range(config.train_num_episodes):
        train_loss = 0
        dev_loss = 0
        optimizer = optim.Adam(mdpnet_msepi.parameters(), lr=lr, weight_decay=config.weight_decay)
        for i_batch in range(config.train_num_batches):
            train_loss_batch = mdpmodel_train(memory, mdpnet_msepi, optimizer, 2, config)
            dev_loss_batch = mdpmodel_test(dev_memory, mdpnet_msepi, 2, config)
            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        if (i_episode + 1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}'
                  .format(i_episode + 1, train_loss, dev_loss))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        else:
            lr *= config.lr_decay

    print('Learn our mdp model')
    best_train_loss = 100
    lr = config.lr
    for i_episode in range(config.train_num_episodes):
        train_loss = 0
        dev_loss = 0
        optimizer = optim.Adam(mdpnet.parameters(), lr=lr, weight_decay=config.weight_decay)
        for i_batch in range(config.train_num_batches):
            train_loss_batch = mdpmodel_train(memory, mdpnet, optimizer, 1, config)
            dev_loss_batch = mdpmodel_test(dev_memory, mdpnet, 1, config)
            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        if (i_episode + 1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}'
                  .format(i_episode + 1, train_loss, dev_loss))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        else:
            lr *= config.lr_decay

    print('Learn the baseline mdp model')
    best_train_loss = 100
    lr = config.lr
    for i_episode in range(config.train_num_episodes):
        train_loss = 0
        dev_loss = 0
        optimizer = optim.Adam(mdpnet_unweight.parameters(), lr=lr, weight_decay=config.weight_decay)
        for i_batch in range(config.train_num_batches):
            train_loss_batch = mdpmodel_train(memory, mdpnet_unweight, optimizer, 0, config)
            train_loss = (train_loss * i_batch + train_loss_batch) / (i_batch + 1)
            dev_loss_batch = mdpmodel_test(dev_memory, mdpnet_unweight, 0, config)
            dev_loss = (dev_loss * i_batch + dev_loss_batch) / (i_batch + 1)
        if (i_episode+1) % config.print_per_epi == 0:
            print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}'.format(i_episode+1, train_loss, dev_loss))
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
                                                                     config.tc_test_batch_size)
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
    mdpnet.eval()
    mdpnet_unweight.eval()
    mdpnet_msepi.eval()
    target = np.zeros(config.sample_num_traj)

    init_states = []
    for i_episode in range(config.sample_num_traj):
        env.seed(seedvec[i_episode].item())
        init_states.append(preprocess_state(env.reset(), config.state_dim))
    init_states = torch.cat(init_states)
    mv = rollout_batch(init_states, mdpnet, tc, config.eval_num_rollout, eval_qnet,
                      epsilon=0, action_size=config.action_size, maxlength=config.max_length, config=config)
    mv_bsl = rollout_batch(init_states, mdpnet_unweight, tc, config.eval_num_rollout, eval_qnet,
                          epsilon=0, action_size=config.action_size, maxlength=config.max_length, config=config)
    mv_msepi = rollout_batch(init_states, mdpnet_msepi, tc, config.eval_num_rollout, eval_qnet,
                       epsilon=0, action_size=config.action_size, maxlength=config.max_length, config=config)
    time_now = time.time()
    time_eval = time_now - time_pre
    time_pre = time_now

    V,Q = compute_values(traj_set, mdpnet_msepi, tc, eval_qnet, config, model_type='MDP')
    dr_msepi = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    wdr_msepi = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    sdr_msepi = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    swdr_msepi = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)

    V,Q = compute_values(traj_set, mdpnet, tc, eval_qnet, config, model_type='MDP')
    dr = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    wdr = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    sdr = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    swdr = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)

    V, Q = compute_values(traj_set, mdpnet_unweight, tc, eval_qnet, config, model_type='MDP')
    dr_bsl = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    wdr_bsl = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    sdr_bsl = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    swdr_bsl = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)

    V, Q = compute_values(traj_set, mrdr_q, tc, eval_qnet, config, model_type='Q')
    mrdr_qv = V[:, 0]
    mrdr = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    wmrdr = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    V, Q = compute_values(traj_set, mrdr_q, tc, eval_qnet, config, model_type='Q', soften=True)
    smrdr = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    swmrdr = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)

    V, Q = compute_values(traj_set, mrdrv2_q, tc, eval_qnet, config, model_type='Q')
    mrdrv2_qv = V[:, 0]
    mrdrv2 = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    wmrdrv2 = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    V, Q = compute_values(traj_set, mrdrv2_q, tc, eval_qnet, config, model_type='Q', soften=True)
    smrdrv2 = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    swmrdrv2 = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)

    V, Q = compute_values(traj_set, None, None, eval_qnet, config, model_type='IS')
    ips = importance_sampling(traj_set)
    wis = importance_sampling(traj_set, wis=True)
    sis = importance_sampling(traj_set, wis=False, soften=True)
    swis = importance_sampling(traj_set, wis=True, soften=True)
    pdis = doubly_robust(traj_set, V, Q, config, wis=False, soften=False)
    wpdis = doubly_robust(traj_set, V, Q, config, wis=True, soften=False)
    spdis = doubly_robust(traj_set, V, Q, config, wis=False, soften=True)
    swpdis = doubly_robust(traj_set, V, Q, config, wis=True, soften=True)

    time_now = time.time()
    time_dr = time_now - time_pre
    time_pre = time_now

    for i_episode in range(config.sample_num_traj):
        values = deque()
        for i_trials in range(1):
            env.seed(seedvec[i_episode].item())
            state = preprocess_state(env.reset(), config.state_dim)
            true_state = state
            true_done = False
            true_steps = 0
            true_rewards = 0
            while not true_done:
                true_action = epsilon_greedy_action(true_state, eval_qnet, 0, config.action_size)
                true_next_state, true_reward, true_done, _ = env.step(true_action.item())
                true_state = preprocess_state(true_next_state, config.state_dim)
                true_steps += 1
                true_rewards += true_reward
            values.append(true_rewards)
        target[i_episode] = np.mean(values)
    time_now = time.time()
    time_gt = time_now - time_pre

    print('| Sampling: {:.3f}s | Preprocess for MRDR: {:.3f}s | Learn MRDR: {:.3f}s | Learn MDP: {:.3f}s | Learn TC: {:.3f}s '
          '| Eval MDP: {:.3f}s | Eval DR: {:.3f}s | Eval: {:.3f}s |'
          .format(time_sampling, time_premrdr, time_mrdr, time_mdp, time_tc, time_eval, time_dr, time_gt))
    print('Target policy value:', np.mean(target))
    results = [mv, dr, wdr, sdr, swdr, mv_bsl, dr_bsl, wdr_bsl, sdr_bsl, swdr_bsl,
               mv_msepi, dr_msepi, wdr_msepi, sdr_msepi, swdr_msepi,
               mrdr_qv, mrdr, wmrdr, smrdr, swmrdr, mrdrv2_qv, mrdrv2, wmrdrv2, smrdrv2, swmrdrv2,
               ips, wis, sis, swis, pdis, wpdis, spdis, swpdis]
    return results, target