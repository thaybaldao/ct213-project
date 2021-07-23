from ppo import PPOAlgorithm
from a2c import A2CAlgorithm

train_timesteps = int(2e5)
simulation_timesteps = 1200
n_eval_episodes = 100
log_dir_name = "best_logs/"
algo = "ppo"
# algo = "a2c"

if algo == "ppo":
    ppo_algorithm = PPOAlgorithm(
        train_timesteps,
        simulation_timesteps,
        n_eval_episodes,
        log_dir_name
    )
    best_models_names = ['ppo_9', 'ppo_89']
    for best_model_name in best_models_names:
        model_name = ppo_algorithm.train_best(best_model_name)
        ppo_algorithm.evaluate_best(model_name)
    ppo_models = ppo_algorithm.create_best_results_file()
    if len(ppo_models) > 0:
        ppo_algorithm.record_video(ppo_models[0][2])
    else:
        print('Error: There are no PPO models trained.')

if algo == "a2c":
    a2c_algorithm = A2CAlgorithm(
        train_timesteps,
        simulation_timesteps,
        n_eval_episodes,
        log_dir_name
    )
    best_models_names = ['a2c_9', 'a2c_89']
    for best_model_name in best_models_names:
        model_name = a2c_algorithm.train_best(best_model_name)
        a2c_algorithm.evaluate_best(model_name)
    a2c_models = a2c_algorithm.create_best_results_file()
    if len(a2c_models) > 0:
        a2c_algorithm.record_video(a2c_models[0][2])
    else:
        print('Error: There are no A2C models trained.')
