import gym

from stable_baselines3.common.evaluation import evaluate_policy
import json
from typing import Callable
from hyperparameters import Hyperparameters


class RLAlgorithm:
    def __init__(self):
        # Create environment
        self.env = gym.make('LunarLander-v2')
        self.params = Hyperparameters()
        self.model = None
        self.algo_name = ""
        self.train_timesteps = int(1e5)
        self.simulation_timesteps = int(1e5)
        self.n_eval_episodes = 50
        self.models = []

    def grid_search(self):
        hyp_id = 0
        for learning_idx in range(4):
            for actor_idx in range(3):
                for critic_idx in range(3):
                    hyp_id += 1
                    dictionary, model_name = self.train(learning_idx, actor_idx, critic_idx, hyp_id)
                    self.evaluate(model_name, dictionary)

    def train(self, learning_idx, actor_idx, critic_idx, hyp_id):
        # Choose hyperparameters
        learning_rate, policy_kwargs = self.choose_params(learning_idx, actor_idx, critic_idx)
        # Create the model
        model_name = self.create_model(learning_rate, policy_kwargs, hyp_id)
        # Train the model
        self.model.learn(total_timesteps=self.train_timesteps)
        # Save the model
        self.model.save(model_name)
        return self.create_model_info_dict(learning_rate, policy_kwargs)

    def choose_params(self, learning_idx, actor_idx, critic_idx):
        learning_rate = self.params.create_learning_rate(learning_idx)
        policy_kwargs = self.params.create_policy_args(2, actor_idx, 2, critic_idx)
        return learning_rate, policy_kwargs

    def create_model(self, learning_rate, policy_kwargs, hyp_id):
        raise NotImplementedError('Please implement this method')

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    def create_model_info_dict(self, learning_rate, policy_kwargs):
        return dict(
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
        )

    def evaluate(self, model_name, dictionary):
        self.load_model_from_file(model_name)
        dictionary['mean_reward'], dictionary['std_reward'] = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=self.n_eval_episodes)
        self.saveJSON(model_name, dictionary)
        self.models.append((dictionary['mean_reward'], model_name))

    def load_model_from_file(self, model_name):
        raise NotImplementedError('Please implement this method')

    def saveJSON(self, file_name, dictionary):
        with open(file_name + ".json", "w") as outfile:
            json.dump(dictionary, outfile, indent=4)

    def create_results_file(self):
        self.models.sort(reverse=True)
        f = open("results.txt", "w")
        for model in self.models:
            line = "Model: " + model(1) + " Mean reward: " + str(model(0)) + "\n"
            f.write(line)
        f.close()
        return self.models

    def simulate(self, model_name):
        obs = self.env.reset()
        model = self.load_model_from_file(model_name)
        for i in range(self.simulation_timesteps):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
              obs = self.env.reset()
