from ppo import PPOAlgorithm
from a2c import A2CAlgorithm

# TODO: choose training and evaluating parameters
train_timesteps = int(1e5)
simulation_timesteps = 1200
n_eval_episodes = 30
log_dir_name = "logs/"
env_name = 'LunarLander-v2'
net_size = 2  # how many different number of layers will be used in the actor and critic nets in the grid search
learning_rate_size = 3  # how many different learning rate values will be used in the grid search
gamma_size = 3  # how many different gamma values will be used in the grid search
actor_size = 3  # how many different unit values will be used in the actor net in the grid search
critic_size = 3  # how many different unit values will be used in the critic net in the grid search

# TODO: choose which algorithm you want to train
algo = "ppo"
# algo = "a2c"

if algo == "ppo":
    ppo_algorithm = PPOAlgorithm(
        train_timesteps,
        simulation_timesteps,
        n_eval_episodes,
        log_dir_name,
        env_name,
        net_size,
        learning_rate_size,
        gamma_size,
        actor_size,
        critic_size,
    )
    ppo_algorithm.grid_search()
    ppo_models = ppo_algorithm.create_results_file()
    if len(ppo_models) > 0:
        ppo_algorithm.simulate(ppo_models[0][1])
    else:
        print('Error: There are no PPO models trained.')

if algo == "a2c":
    a2c_algorithm = A2CAlgorithm(
        train_timesteps,
        simulation_timesteps,
        n_eval_episodes,
        log_dir_name,
        env_name,
        net_size,
        learning_rate_size,
        gamma_size,
        actor_size,
        critic_size,
    )
    a2c_algorithm.grid_search()
    a2c_models = a2c_algorithm.create_results_file()
    if len(a2c_models) > 0:
        best_model_name = a2c_models[0][1]
        a2c_algorithm.record_video(a2c_models[0][1])
    else:
        print('Error: There are no a2c models trained.')
