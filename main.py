from ppo import PPOAlgorithm

ppo_algorithm = PPOAlgorithm()
ppo_algorithm.grid_search()
models = ppo_algorithm.create_results_file()
print(models[0][1])
ppo_algorithm.simulate(models[0][1])
