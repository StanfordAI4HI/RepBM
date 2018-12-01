# Representation Balancing MDPs for Off-Policy Policy Evaluation

This repository contains an implementation of the representation balancing MDP (RepBM) OPPE estimator 
in paper [Representation Balancing MDPs for Off-Policy Policy Evaluation](https://arxiv.org/abs/1805.09044). 
The code is implemented in Python 3.6 using pytorch 0.4.1 and numpy 1.14.2.

## RepBM Model
We implemented the RepBM using neural networks as function approximator and focus on deterministic
transition case (or stochastic transition in tabular state space). The model of RepBM is defined in 
```src/models.py```. The core components of the RepBM algorithm such as the loss functin is implemented
 in ``mdpmodel_train`` function in ``train_pipeline.py``. Hyper-parameters of the nn model is specified
  in ```src/config.py```.

## Example
Example domains from the experiment section in the paper in included in this repository.
### CartPole and MountainCar
We use the CartPole-v0 domain and MountainCar-v0 domain in OpenAI Gym. 

An example of running the 
experiment in CartPole domain:
```sh
$ python qlearning_cartpole.py
$ python main_cartpole.py
```

```qlearning_cartpole.py``` will learn a near-optimal value function and save it in directory 
```target_policies```. The greedy policy based on learned value function will be used as evaluation 
policy and the epsilon-greedy policy will serve as behavior policy. As an example, we also include the policies used in 
the experiment section of the paper, in ```target_policies```.

To run experiment across several different values of hyper-parameter alpha:

```sh
$ python parameter_searching_cartpole.py
```

### HIV simulator
The HIV simulator and the a FQI learning algorithm is implemented in directory ```hiv_domain```.
 The code modified based on RLPy and Harvard DTAK group's implementation. To run the experiment:
 
 
```sh
$ python hiv_domain/qlearning_hiv.py
$ python compute_hiv_standardization_data.py
$ python main_hiv.py
$ python analyze_hiv.py
```

## Citation
```bash
@incollection{NIPS2018_7530,
title = {Representation Balancing MDPs for Off-policy Policy Evaluation},
author = {Liu, Yao and Gottesman, Omer and Raghu, Aniruddh and Komorowski, Matthieu and Faisal, Aldo A and Doshi-Velez, Finale and Brunskill, Emma},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {2645--2654},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7530-representation-balancing-mdps-for-off-policy-policy-evaluation.pdf}
}
```