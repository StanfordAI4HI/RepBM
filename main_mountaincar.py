import numpy as np
import torch
import gym
from src.models import QNet
from src.config import mountaincar_config
from src.train_pipeline import train_pipeline
from src.utils import load_qnet, error_info
from collections import deque

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    config = mountaincar_config
    eval_qnet = QNet(config.state_dim,config.dqn_hidden_dims,config.action_size)
    load_qnet(eval_qnet,filename='qnet_mc.pth.tar')
    eval_qnet.eval()

    methods = ['Model', 'DR', 'WDR', 'Soft DR', 'Soft WDR',
               'Model Bsl', 'DR Bsl', 'WDR Bsl', 'Soft DR Bsl', 'Soft WDR Bsl',
               'Model MSE', 'DR MSE', 'WDR MSE', 'Soft DR MSE', 'Soft WDR MSE',
               'MRDR Q', 'MRDR', 'WMRDR', 'Soft MRDR', 'Soft WMRDR',
               'MRDR-w Q', 'MRDR-w', 'WMRDR-w', 'Soft MRDR-w', 'Soft WMRDR-w',
               'IS', 'WIS', 'Soft IS', 'Soft WIS', 'PDIS', 'WPDIS', 'Soft PDIS', 'Soft WPDIS']
    num_method = len(methods)
    max_name_length = len(max(methods,key=len))

    mse = [deque() for method in methods]
    ind_mse = [deque() for method in methods]

    for i_run in range(config.N):
        print('Run: {}'.format(i_run+1))
        results, target = train_pipeline(env,config,eval_qnet)
        for i_method in range(num_method):
            #print(results[i_method])
            mse_1, mse_2 = error_info(results[i_method], target, methods[i_method].ljust(max_name_length))
            mse[i_method].append(mse_1)
            ind_mse[i_method].append(mse_2)

    mse_table = np.zeros((num_method,4))
    print('Average result over {} runs:'.format(config.N))
    for i in range(num_method):
        print('{}: Root mse of mean is {:.3e}±{:.2e}, root mse of individual is {:.3e}±{:.2e}'
              .format(methods[i].ljust(max_name_length), np.sqrt(np.mean(mse[i])), np.sqrt(np.std(mse[i])),
                      np.sqrt(np.mean(ind_mse[i])), np.sqrt(np.std(ind_mse[i]))))
        mse_table[i, 0] = np.sqrt(np.mean(mse[i]))
        mse_table[i, 1] = np.sqrt(np.std(mse[i]))
        mse_table[i, 2] = np.sqrt(np.mean(ind_mse[i]))
        mse_table[i, 3] = np.sqrt(np.std(ind_mse[i]))
    np.savetxt('results/result_mountaincar.txt', mse_table, fmt='%.3e', delimiter=',')
