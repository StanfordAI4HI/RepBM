from src.config import hiv_config
from hiv_domain.fittedQiter import FittedQIteration
from sklearn.externals import joblib
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    
    ep_reward = []
    config = hiv_config
    rewards = []

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
    for i_episode in range(config.sample_num_traj):
        if i_episode % 10 == 0:
            print("{} trajectories generated".format(i_episode))
        #episode = env.run_episode(eps=config.behavior_eps, track=True)
        episode = env.run_episode(eps=0, track=True)
        G = 0
        for idx_step in range(config.max_length):
            G += episode[idx_step][2]*config.gamma**idx_step
        ep_reward.append(G)
        rewards += ([episode[i][2] for i in range(len(episode))])

standardization_data = dict()
standardization_data[config.ins] = dict()
standardization_data[config.ins]["reward_mean"] = np.mean(rewards)
standardization_data[config.ins]["reward_std"] = np.std(rewards)
with open("hiv_domain/standardization_data.pkl", "wb") as f:
   pickle.dump(standardization_data, f)
