from hiv_domain.fittedQiter import FittedQIteration
from hiv_domain.hiv_simulator.hiv import HIVTreatment as model
import pickle
from sklearn.externals import joblib
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# stable_ins = [4,14, 16, 18, 20, 32,36]
# test_ins = [43, 49,52,53,54]
with open('hiv_simulator/hiv_preset_hidden_params','rb') as f:
    preset_hidden_params = pickle.load(f, encoding='latin1')
ins = 20
qiter = FittedQIteration(perturb_rate = 0.03,preset_params =preset_hidden_params[ins],gamma = 0.98,ins=ins)

if True:
    print('Learning the tree')
    qiter.updatePlan()
    joblib.dump(qiter.tree, 'extra_tree_gamma_ins' + str(qiter.ins) + '.pkl')
else:
    print('Load the tree from file')
    qiter.tree = joblib.load('extra_tree_gamma_ins20.pkl')
ep = qiter.run_episode(eps = 0.05,track = True)

ys=['logT1','logT2','logT1*','logT2*','logV','logE']

for k in range(6):
    plt.subplot(3,2,k+1)
    plt.plot(np.arange(200),[d[0][k] for d in ep])
    plt.ylabel(ys[k])
    plt.savefig('traj.png')
plt.show()
