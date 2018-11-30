import os
import numpy as np
import random
import torch
from torch.autograd import Variable

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def save_qnet(state, checkpoint='target_policies',filename='qnet.pth.tar'):
    # From https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}"
              .format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)


def load_qnet(model, checkpoint='target_policies',filename='qnet.pth.tar'):
    # From https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(filepath):
        raise("No best model in path {}".format(checkpoint))
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint


def preprocess_state(state, input_dim):
    return Tensor(np.reshape(state, [1, input_dim]))


def preprocess_action(action):
    return LongTensor([[action]])


def preprocess_reward(reward):
    return FloatTensor([reward])


def rescale_state(state, rescale=1):
    return state*rescale


def select_action_random(action_size):
    return LongTensor([[random.randrange(action_size)]])


def select_maxq_action(state_tensor, qnet):
    return qnet.forward(state_tensor.type(FloatTensor)).detach().max(1)[1].view(-1, 1)


def epsilon_greedy_action_batch(state_tensor, qnet, epsilon, action_size):
    batch_size = state_tensor.size()[0]
    sample = np.random.random([batch_size,1])
    greedy_a = qnet.forward(
            state_tensor.type(FloatTensor)).detach().max(1)[1].view(-1, 1)
    random_a = LongTensor(np.random.random_integers(0,action_size-1,(batch_size,1)))
    return (sample < epsilon)*random_a + (sample >= epsilon)*greedy_a


def epsilon_greedy_action(state_tensor, qnet, epsilon, action_size, q_values=None):
    if q_values is None:
        q_values = qnet.forward(Variable(state_tensor, volatile=True).type(FloatTensor)).detach()
    sample = random.random()
    if sample > epsilon:
        return q_values.max(1)[1].view(-1, 1)
    else:
        return LongTensor([[random.randrange(action_size)]])


def epsilon_greedy_action_prob(state_tensor, qnet, epsilon, action_size, q_values = None):
    if q_values is None:
        q_values = qnet.forward(Variable(state_tensor, volatile=True).type(FloatTensor)).detach()
    max_action = q_values.max(1)[1].view(1, 1).item()
    prob = FloatTensor(1,action_size)
    prob.fill_(epsilon/action_size)
    prob[0,max_action] = 1-epsilon+epsilon/action_size
    return prob


def restrict_state_region(state_tensor):
    flag = abs(state_tensor[0,0].item()) < 2.2 and abs(state_tensor[0,2].item()) < 0.2
    return flag


def friendly_epsilon_greedy_action(state_tensor, qnet, epsilon, action_size, q_values=None):
    if q_values is None:
        q_values = qnet.forward(Variable(state_tensor, volatile=True).type(FloatTensor)).detach()
    sample = random.random()
    if restrict_state_region(state_tensor):
        sample = 1
    if sample > epsilon:
        return q_values.max(1)[1].view(-1, 1)
    else:
        return LongTensor([[random.randrange(action_size)]])


def friendly_epsilon_greedy_action_prob(state_tensor, qnet, epsilon, action_size, q_values = None):
    if q_values is None:
        q_values = qnet.forward(Variable(state_tensor, volatile=True).type(FloatTensor)).detach()
    max_action = q_values.max(1)[1].view(1, 1).item()
    prob = FloatTensor(1,action_size)
    if restrict_state_region(state_tensor):
        prob.fill_(0)
        prob[0, max_action] = 1
    else:
        prob.fill_(epsilon / action_size)
        prob[0, max_action] = 1 - epsilon + epsilon / action_size
    return prob


def soft_policy(q_values, alpha, beta):
    u = random.random()-0.5
    threshold = alpha + u*beta
    action = q_values.max(1)[1].view(1, 1)
    prob = FloatTensor(1, 2)
    prob[0, action.item()] = threshold
    prob[0, 1-action.item()] = 1-threshold

    sample = random.random()
    if sample < threshold:
        return action, prob
    else:
        return 1-action, prob


def learning_rate_update(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def weighted_mse_loss(input, target, weights):
    out = ((input-target)**2)
    if len(list(out.size())) > 1:
        out = out.mean(1)
    out = out * weights #.expand_as(out)
    loss = out.mean(0)
    return loss


def mmd_lin(rep1, rep2, p=0.5):
    mean1 = rep1.mean(0)
    mean2 = rep2.mean(0)
    mmd = ((2.0*p*mean1 - 2.0*(1.0-p)*mean2)**2).sum(0)
    return mmd


def mmd_rbf(rep1, rep2, sigma = 0.1):
    K12 = torch.exp(-pdist(rep1, rep2)/sigma**2)
    K11 = torch.exp(-pdist(rep1, rep1) / sigma ** 2)
    K22 = torch.exp(-pdist(rep2, rep2) / sigma ** 2)

    m1 = rep1.size()[0]
    m2 = rep2.size()[0]
    if m1 > 1:
        d11 = 1 / (m1 * (m1-1)) * (torch.sum(K11)-m1)
    else:
        d11 = 0
    if m2 > 1:
        d22 = 1 / (m2 * (m2-1)) * (torch.sum(K22)-m2)
    else:
        d22=0
    d12 = - 2.0 / (m1 * m2) * torch.sum(K12)
    mmd = d11 + d22 + d12

    return mmd


def pdist(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    # From https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py
    M = -2*torch.matmul(X,torch.transpose(Y,0,1))
    sqx = torch.sum(X**2,1,keepdim=True)
    sqy = torch.sum(Y**2,1,keepdim=True)
    D = M + torch.transpose(sqy,0,1) + sqx
    return D


def error_info(estm, target, method_name):
    print('{}: Mean value is {:.3f}±{:.3e}, mse of mean is {:.3e}, mse of individual is {:.3e}'
          .format(method_name, np.mean(estm), np.std(estm),
                  (np.mean(estm) - np.mean(target)) ** 2, np.mean((estm - target) ** 2)))
    return (np.mean(estm) - np.mean(target)) ** 2, np.mean((estm - target) ** 2)

