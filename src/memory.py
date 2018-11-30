from collections import namedtuple
import random
import numpy as np

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done', 'isweight'))
MyTransition = namedtuple('MyTransition',('state', 'action', 'next_state', 'reward', 'done'
                                          , 'isweight', 'acc_isweight', 'time', 'factual', 'last_factual'
                                          , 'acc_soft_isweight', 'soft_isweight', 'soft_pie', 'pie', 'pib'))
MRDRTransition = namedtuple('MRDRTransition',('state', 'action', 'next_state', 'reward', 'done'
                                              ,'isweight', 'acc_isweight', 'time', 'factual', 'last_factual'
                                              , 'acc_soft_isweight', 'soft_isweight', 'soft_pie', 'pie', 'pib', 'acc_reward', 'wacc_reward'))

# This is a trajectory set, which is used for IS
class TrajectorySet(object):
    def __init__(self, args):
        self.trajectories = []
        self.position = -1
        self.max_len = args.max_length

    def new_traj(self):
        traj = []
        self.trajectories.append(traj)
        self.position += 1

    def push(self, *args):
        self.trajectories[self.position].append(MyTransition(*args))

    def __len__(self):
        return len(self.trajectories)

# This is the data structure we use to train our MDPnet
# The trick is we store all transition tuples on the same step together, so that when we sample from dataset,
#   we guarantee that the same mini-batch is on the sample time step, which allow us to compute IPM more efficiently
class SampleSet(object):
    def __init__(self, args):
        self.max_len = args.max_length
        self.num_episode = 0
        self.factual = np.zeros(self.max_len)
        self.u = np.zeros(self.max_len) # u_{0:t}
        self.memory = [[] for h in range(self.max_len)]
        self.terminal = []

    def push(self, *args):
        #print(n_steps)
        t = MyTransition(*args)
        self.memory[t.time].append(t)
        if t.factual[0]==1:
            self.factual[t.time] += 1
        if t.done and t.time < self.max_len-1:
            self.terminal.append(t)

    def update_u(self):
        self.num_episode = len(self.memory[0])
        self.u = self.factual/self.num_episode

    def flatten(self):
        self.allsamples = [item for sublist in self.memory for item in sublist]

    def sample(self, batch_size):
        while True:
            time = random.randint(0,self.max_len-1)
            if len(self.memory[time]) >= batch_size:
                return random.sample(self.memory[time], batch_size)

    def sample_terminal(self, batch_size):
        if len(self.terminal) >= batch_size:
            return random.sample(self.terminal, batch_size)
        else:
            return self.terminal

    def flatten_sample(self, batch_size):
        return random.sample(self.allsamples, batch_size)

    def sample_given_t(self, batch_size, time):
        if len(self.memory[time]) >= batch_size:
            return random.sample(self.memory[time], batch_size)
        else:
            return self.memory[time]

    def __len__(self):
        return len(self.memory[0])

# This is the replay memory to train dqn for cartpole - because we need to learn an eval policy at first
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
