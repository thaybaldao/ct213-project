import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import json
from typing import Callable
from hyperparameters import Hyperparameters
from stable_baselines3.common.monitor import Monitor
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class RLAlgorithm:
    def __init__(self, train_timesteps, simulation_timesteps, n_eval_episodes, log_dir_name, env_name='LunarLander-v2', net_size=2, learning_rate_size=1, gamma_size=1,
                 actor_size=1, critic_size=1):
        self.params = Hyperparameters()
        self.algo_name = ""
        self.train_timesteps = train_timesteps
        self.simulation_timesteps = simulation_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.models = []
        self.best_models = []
        self.net_size = net_size
        self.learning_rate_size = learning_rate_size
        self.gamma_size = gamma_size
        self.actor_size = actor_size
        self.critic_size = critic_size
        self.log_dir_name = log_dir_name
        self.log_dir = os.getcwd() + "/" + self.log_dir_name
        os.makedirs(self.log_dir, exist_ok=True)
        self.env_name = env_name
        self.env = None
        self.models_dir_name = self.log_dir_name + "models/"
        self.models_dir = os.getcwd() + "/" + self.models_dir_name

    def grid_search(self):
        hyp_id = 0
        for net_layers in range(2, 2 + self.net_size):
            for learning_idx in range(self.learning_rate_size):
                for gamma_idx in range(self.gamma_size):
                    for actor_idx in range(self.actor_size):
                        for critic_idx in range(self.critic_size):
                            hyp_id += 1
                            self.env = Monitor(gym.make(self.env_name))
                            dictionary, model_name = self.train(learning_idx, gamma_idx, actor_idx, critic_idx,
                                                                net_layers, hyp_id)
                            self.evaluate(model_name, dictionary)

    def plot_results(self, file_name):
        sns.set_theme()
        qualitative_colors = sns.color_palette("bright")
        fig_dir = self.log_dir + "figures/"
        os.makedirs(fig_dir, exist_ok=True)
        plt.xlim(0, self.env.get_total_steps())
        plt.ylabel('Rewards')
        plt.xlabel('Timesteps')
        x = np.cumsum(self.env.get_episode_lengths())
        plt.plot(x, np.zeros(len(x)), color="k", lw=1)
        plt.plot(x, self.env.get_episode_rewards(), color=qualitative_colors[0], lw=1)
        plt.savefig(fig_dir + file_name, format="png", bbox_inches="tight")
        plt.clf()

    def train(self, learning_idx, gamma_idx, actor_idx, critic_idx, net_layers, hyp_id):
        # Choose hyperparameters
        learning_rate, gamma, policy_kwargs, policy_info = self.choose_params(learning_idx, gamma_idx, actor_idx,
                                                                              critic_idx, net_layers)
        # Create the model
        model, model_name = self.create_model(learning_rate, gamma, policy_kwargs, hyp_id)
        # Train the model
        model.learn(total_timesteps=self.train_timesteps)
        # Plot the results
        self.plot_results("{}_{}.png".format(self.algo_name, hyp_id))
        # Save the model
        model.save(model_name)
        model_dict = self.create_model_info_dict(model_name, learning_rate, gamma, policy_info)
        return model_dict, model_name

    def choose_params(self, learning_idx, gamma_idx, actor_idx, critic_idx, net_layers):
        learning_rate = self.params.create_learning_rate(learning_idx)
        gamma = self.params.create_gamma(gamma_idx)
        policy_kwargs, policy_info = self.params.create_policy_kwargs(net_layers, actor_idx, net_layers, critic_idx)
        return learning_rate, gamma, policy_kwargs, policy_info

    def create_model(self, learning_rate, gamma, policy_kwargs, hyp_id):
        raise NotImplementedError('Please implement this method')

    @staticmethod
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return func

    @staticmethod
    def create_model_info_dict(model_name, learning_rate, gamma, policy_info):
        return {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "policy_kwargs": policy_info
        }

    def evaluate(self, model_name, dictionary):
        model = self.load_model_from_file(model_name)
        dictionary['mean_reward'], dictionary['std_reward'] = evaluate_policy(model, model.get_env(),
                                                                              n_eval_episodes=self.n_eval_episodes)
        self.save_json(model_name, dictionary)
        self.models.append((dictionary['mean_reward'], model_name))

    def load_model_from_file(self, model_name):
        raise NotImplementedError('Please implement this method')

    def save_json(self, model_name, dictionary):
        os.makedirs(self.models_dir, exist_ok=True)
        with open(self.models_dir_name + model_name + ".json", "w") as outfile:
            json.dump(dictionary, outfile, indent=4)

    def create_results_file(self):
        self.models.sort(reverse=True)
        f = open("results_" + self.algo_name + ".txt", "w")
        for model in self.models:
            line = "Model: " + model[1] + " Mean reward: " + str(model[0]) + "\n"
            f.write(line)
        f.close()
        return self.models

    def simulate(self, model_name):
        self.env = Monitor(gym.make(self.env_name))
        obs = self.env.reset()
        model = self.load_model_from_file(model_name)
        for i in range(self.simulation_timesteps):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()

    def record_video(self, model_name):
        video_length = self.simulation_timesteps
        record_env = DummyVecEnv([lambda: gym.make(self.env_name)])
        obs = record_env.reset()
        record_env = VecVideoRecorder(record_env, 'logs/videos/',
                                      record_video_trigger=lambda x: x == 0, video_length=video_length,
                                      name_prefix=model_name)
        record_env.reset()
        model = self.load_model_from_file(model_name)
        for i in range(video_length + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = record_env.step(action)
        # Save the video
        record_env.close()

    def train_best(self, model_name):
        self.env = Monitor(gym.make(self.env_name))
        # Choose hyperparameters
        learning_rate, gamma, policy_kwargs = self.choose_params_from_file(model_name)
        hyp_id = self.parse_hyp_id(model_name)
        # Create the model
        model, model_name = self.create_model(learning_rate, gamma, policy_kwargs, hyp_id)
        # Train the model
        model.learn(total_timesteps=self.train_timesteps)
        # Plot the results
        self.plot_results("{}_{}.png".format(self.algo_name, hyp_id))
        # Save the model
        model.save(model_name)
        return model_name

    def choose_params_from_file(self, model_name):
        dictionary = self.read_json(model_name)
        learning_rate = dictionary['learning_rate']
        gamma = dictionary['gamma']
        policy_kwargs = self.params.create_policy_kwargs_from_dict(dictionary['policy_kwargs'])
        return learning_rate, gamma, policy_kwargs

    def parse_hyp_id(self, model_name):
        split_model_name = model_name.split("_")
        return int(split_model_name[1]) + 1000  # + 1000 to avoid overriding grid_search results

    def read_json(self, model_name):
        self.models_dir_name = "logs/models/"
        self.models_dir = os.getcwd() + "/" + self.models_dir_name
        os.makedirs(self.models_dir, exist_ok=True)
        with open(self.models_dir_name + model_name + ".json") as json_file:
            dictionary = json.load(json_file)
        print(dictionary)
        return dictionary

    def evaluate_best(self, model_name):
        model = self.load_model_from_file(model_name)
        mean_reward, std_reward = evaluate_policy(model, model.get_env(),
                                                  n_eval_episodes=self.n_eval_episodes)
        self.best_models.append(mean_reward, std_reward, model_name)

    def create_best_results_file(self):
        self.best_models.sort(reverse=True)
        f = open("best_results_" + self.algo_name + ".txt", "w")
        for model in self.models:
            line = "Model: " + model[2] + " Mean reward: " + str(model[0]) + " Std reward: " + str(model[1]) + "\n"
            f.write(line)
        f.close()
        return self.models
