class cartpole_config():
    # domain parameters
    state_dim = 4
    action_size = 2
    gamma = 1.0
    n_win_ticks = 195
    max_length = 200
    oracle_reward = 1
    rescale = [[1, 1, 1, 1]]

    # q training parameters
    dqn_batch_size = 64
    dqn_hidden_dims = [24,48]
    dqn_num_episodes = 2000
    buffer_capacity = 10000
    dqn_alpha = 0.01
    dqn_alpha_decay = 0.01
    dqn_epsilon = 1.0
    dqn_epsilon_min = 0.01
    dqn_epsilon_decay = 0.995
    sample_capacity = 200000

    # model parameters
    sample_num_traj = 1024
    train_num_traj = 900
    dev_num_traj = 124
    transition_input_dims = 4
    rep_hidden_dims = [16] # The last dim is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []
    terminal_hidden_dims = [32,32]
    behavior_epsilon = 0.2
    eval_epsilon = 0.0

    # model training parameter
    print_per_epi = 10
    train_num_episodes = 100
    train_num_batches = 100
    train_batch_size = 64
    test_batch_size = 16
    tc_num_episode = 100
    tc_num_batches = 100
    tc_batch_size = 64
    tc_test_batch_size = 16
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.01
    weight_decay = 0

    # MRDR parameter
    soften_epsilon = 0.02
    mrdr_lr = 0.01
    mrdr_num_episodes = 100
    mrdr_num_batches = 100
    mrdr_batch_size = 1000
    mrdr_test_batch_size = 100
    mrdr_hidden_dims = [32]

    eval_num_traj = 1000
    eval_num_rollout = 1
    N = 100
    MAX_SEED = 1000000


class cartpole_test_config():
    # domain parameters
    state_dim = 4
    action_size = 2
    gamma = 1.0
    n_win_ticks = 195
    max_length = 200
    oracle_reward = 1
    rescale = [[1, 1, 1, 1]]

    # q training parameters
    dqn_batch_size = 64
    dqn_hidden_dims = [24,48]
    dqn_num_episodes = 2000
    buffer_capacity = 10000
    dqn_alpha = 0.01
    dqn_alpha_decay = 0.01
    dqn_epsilon = 1.0
    dqn_epsilon_min = 0.01
    dqn_epsilon_decay = 0.995
    sample_capacity = 200000

    # model parameters
    sample_num_traj = 10 #1024
    train_num_traj = 8 #900
    dev_num_traj = 2 #124
    transition_input_dims = 4
    rep_hidden_dims = [16] # The last dim is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []
    terminal_hidden_dims = [32,32]
    behavior_epsilon = 0.2
    eval_epsilon = 0.0

    # model training parameter
    print_per_epi = 10
    train_num_episodes = 100
    train_num_batches = 100
    train_batch_size = 4 #64
    test_batch_size = 2 #16
    tc_num_episode = 100
    tc_num_batches = 100
    tc_batch_size = 2 #64
    tc_test_batch_size = 2 #16
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.01
    weight_decay = 0 #or 0.00005

    # MRDR parameter
    soften_epsilon = 0.02
    mrdr_lr = 0.01
    mrdr_num_episodes = 20
    mrdr_num_batches = 50
    mrdr_batch_size = 4 #1000
    mrdr_test_batch_size = 2 #100
    mrdr_hidden_dims = [32]

    eval_num_traj = 1000
    eval_num_rollout = 1
    N = 2
    MAX_SEED = 1000000


class mountaincar_config():
    # domain parameters
    state_dim = 2
    action_size = 3
    gamma = 0.99
    max_length = 200
    oracle_reward = -1
    rescale = [[1, 10]]

    # q training parameters
    dqn_batch_size = 256
    dqn_hidden_dims = [100]
    dqn_num_episodes = 10000
    buffer_capacity = 20000
    dqn_alpha = 0.01
    dqn_epsilon = 0.5
    dqn_epsilon_min = 0.05
    dqn_epsilon_decay = 0.9995
    sample_capacity = 200000
    target_update = 10

    # model parameters
    sample_num_traj = 1024
    train_num_traj = 900
    dev_num_traj = 124
    transition_input_dims = 4
    rep_hidden_dims = [16] # The last dim is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []
    terminal_hidden_dims = [32,32]
    behavior_epsilon = 0.2
    eval_epsilon = 0.0

    # model training parameter
    print_per_epi = 10
    train_num_episodes = 100
    train_num_batches = 50
    train_batch_size = 16
    test_batch_size = 16
    tc_num_episode = 100
    tc_num_batches = 50
    tc_batch_size = 16
    tc_test_batch_size = 16
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.1
    weight_decay = 0.00005

    # MRDR parameter
    soften_epsilon = 0.02
    mrdr_lr = 0.01
    mrdr_num_episodes = 100
    mrdr_num_batches = 50
    mrdr_batch_size = 1000
    mrdr_test_batch_size = 100
    mrdr_hidden_dims = [32]

    eval_num_traj = 1000
    eval_num_rollout = 1
    N = 100
    MAX_SEED = 1000000


class mountaincar_test_config():
    # domain parameters
    state_dim = 2
    action_size = 3
    gamma = 0.99
    max_length = 200
    oracle_reward = -1
    rescale = [[1, 10]]

    # q training parameters
    dqn_batch_size = 256
    dqn_hidden_dims = [100]
    dqn_num_episodes = 10000
    buffer_capacity = 20000
    dqn_alpha = 0.01
    #dqn_alpha_decay = 0.01
    dqn_epsilon = 0.5
    dqn_epsilon_min = 0.05
    dqn_epsilon_decay = 0.9995
    sample_capacity = 200000
    target_update = 10

    # model parameters
    sample_num_traj = 10 #1024
    train_num_traj = 8 #900
    dev_num_traj = 2 #124
    transition_input_dims = 4
    rep_hidden_dims = [16] # The last dim is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []
    terminal_hidden_dims = [32,32]
    behavior_epsilon = 0.2
    eval_epsilon = 0.0

    # model training parameter
    print_per_epi = 10
    train_num_episodes = 100
    train_num_batches = 50
    train_batch_size = 4 #16
    test_batch_size = 2 #16
    tc_num_episode = 100
    tc_num_batches = 50
    tc_batch_size = 2 #16
    tc_test_batch_size = 2 #16
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.1
    weight_decay = 0.00005

    # MRDR parameter
    soften_epsilon = 0.02
    mrdr_lr = 0.01
    mrdr_num_episodes = 20
    mrdr_num_batches = 50
    mrdr_batch_size = 2 #1000
    mrdr_test_batch_size = 2 #100
    mrdr_hidden_dims = [32]

    eval_num_traj = 1000
    eval_num_rollout = 1
    N = 2
    MAX_SEED = 1000000


class hiv_config():

    # domain parameters
    state_dim = 6
    action_size = 4
    gamma = 0.98
    max_length = 200

    # model parameters
    sample_num_traj = 40
    train_num_traj = 45
    dev_num_traj = 5
    rep_hidden_dims = [64, 64] # The last layer is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []

    print_per_epi = 10
    train_num_episodes = 100
    train_num_batches = 100
    train_batch_size = 40
    test_batch_size = 5
    train_traj_batch_size = 4
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.1

    # eval_num_traj = 1000
    eval_num_rollout = 1
    eval_pib_num_rollout = 100

    N = 10

    fix_data = False

    behavior_eps = 0.05
    standardize_rewards = True
    ins = 20