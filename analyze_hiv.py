"""
Loads saved HIV results and displays a comparison between the resutls of the 
representation balancing MDP and a vanilla MDP
"""

import numpy as np
import matplotlib.pyplot as plt
import os

base_fname = "results/data_hiv"
fname = base_fname + ".npz"
idx_file = 1
while os.path.isfile(fname):
    loaded_data = np.load(fname)    
    if idx_file == 1:
        estm_list = list(loaded_data["estm_list"])
        estm_bsl_list = list(loaded_data["estm_bsl_list"])
        ips_list = list(loaded_data["ips_list"])
        pdis_list = list(loaded_data["pdis_list"])
        wpdis_list = list(loaded_data["wpdis_list"])
    else:
        estm_list += list(loaded_data["estm_list"])
        estm_bsl_list += list(loaded_data["estm_bsl_list"])
        ips_list += list(loaded_data["ips_list"])
        pdis_list += list(loaded_data["pdis_list"])
        wpdis_list += list(loaded_data["wpdis_list"])
    idx_file += 1
    fname = base_fname + str(idx_file) + ".npz"
        
        
true_value = 5.9*10**9

print('estm err : ',
      np.mean((np.array(estm_list) - true_value) ** 2)**0.5 /
      true_value)
print('estm err ste : ',
      np.std((np.array(estm_list) - true_value) ** 2)**0.5 
      / len(estm_list)**0.5 / true_value)
print('estm_bsl err : ',
      np.mean((np.array(estm_bsl_list) - true_value) ** 2)**0.5 /
      true_value)
print('estm_bsl err ste : ',
      np.std((np.array(estm_bsl_list) - true_value) ** 2)**0.5
      / len(estm_list)**0.5 / true_value)
print('ips err : ',
      np.mean((np.array(ips_list) - true_value) ** 2)**0.5 /
      true_value)
print('ips err ste : ',
      np.std((np.array(ips_list) - true_value) ** 2)**0.5
      / len(estm_list)**0.5 / true_value)
print('pdis err : ',
      np.mean((np.array(pdis_list) - true_value) ** 2)**0.5 /
      true_value)
print('pdis err ste : ',
      np.std((np.array(pdis_list) - true_value) ** 2)**0.5
      / len(estm_list)**0.5 / true_value)
print('wpdis err : ',
      np.mean((np.array(wpdis_list) - true_value) ** 2)**0.5 /
      true_value)
print('wpdis err ste : ',
      np.std((np.array(wpdis_list) - true_value) ** 2)**0.5
      / len(estm_list)**0.5 / true_value)

normalize = True
bins_num = 'auto'
est_us_hy, est_us_hx = np.histogram(estm_list, bins_num, normed=normalize)
est_bsl_hy, est_bsl_hx = np.histogram(estm_bsl_list, bins_num, normed=normalize)

plt.plot((est_us_hx[:-1]+est_us_hx[1:])/2, est_us_hy, label = "our model")
plt.plot((est_bsl_hx[:-1]+est_bsl_hx[1:])/2, est_bsl_hy, label = "baseline model")
plt.plot([true_value, true_value], [0, max(est_us_hy)], label = "\"true\" value")
plt.xlabel("value estimate")
plt.legend()
plt.savefig('results/hiv_figures/tempfig_1.png')