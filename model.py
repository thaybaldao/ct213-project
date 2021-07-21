import gym
import torch as th

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import json
from typing import Callable

class RLAlgorithm:
    def __init__(self):
        # Create environment
        self.env = gym.make('LunarLander-v2')
        self.hyperparameters = Hyperparameters()
        self.model = None
        self.algo_name = ""
        self.train_timesteps = int(1e5)
        self.n_eval_episodes = 50

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return func

    def create_model(self):
        raise NotImplementedError('Please implement this method')

    def create_model_name(self, hyp_id):
        return self.algo_name + "_" + str(hyp_id)

    def train(self, learning_idx, actor_idx, critic_idx, model_name):
        # Choose hyperparameters
        learning_rate, policy_kwargs = self.choose_hyperparameters(learning_idx, actor_idx, critic_idx)
        # Create the model
        self.create_model(learning_rate, policy_kwargs)
        # Train the model
        self.model.learn(total_timesteps=self.train_timesteps)
        # Save the model
        self.model.save(model_name)
        return self.create_model_info_dict(learning_rate, policy_kwargs)

    def evaluate(self, model_name, dictionary):
        self.load_model_from_file(model_name)
        dictionary['mean_reward'], dictionary['std_reward'] = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=self.n_eval_episodes)
        self.saveJSON(model_name, dictionary)

    def create_model_info_dict(self, learning_rate, policy_kwargs):
        return dict(
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
        )

    def saveJSON(self, file_name, dictionary):
        with open(file_name + ".json", "w") as outfile:
            json.dump(dictionary, outfile, indent=4)

    def load_model_from_file(self):
        raise NotImplementedError('Please implement this method')

    def simulate(self, timesteps):
        # Enjoy trained agent
        obs = self.env.reset()
        for i in range(1000):
            action, _state = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
              obs = self.env.reset()

    def grid_search(self):
        hyp_id = 0
        for learning_idx in range(4):
            for actor_idx in range(3):
                for critic_idx in range(3):
                    hyp_id += 1
                    model_name = self.create_model_name(hyp_id)
                    dictionary = self.train(learning_idx, actor_idx, critic_idx, model_name)
                    self.evaluate(model_name, dictionary)

    def choose_hyperparameters(self, learning_idx, actor_idx, critic_idx):
        learning_rate = self.hyperparameters.create_learning_rate(learning_idx)
        policy_kwargs = self.hyperparameters.create_policy_args(2, actor_idx, 2, critic_idx)
        return learning_rate, policy_kwargs, id


class PPOAlgorithm(RLAlgorithm):
    def create_model(self, learning_rate, policy_kwargs):
        self.algo_name = "ppo"
        # Create the agent
        self.model = PPO("MlpPolicy", self.env, learning_rate=learning_rate, policy_kwargs=policy_kwargs, verbose=1)

    def load_model_from_file(self):
        self.model = PPO.load(self.model_name, env=self.env)

class Hyperparameters:
    def create_learning_rate(self, idx):
        learning_rate = 0.001
        return learning_rate/(2**idx)

    def create_net(self, net_size, idx):
        net = []
        for i in range(net_size):
            net.append(2**(5+idx))
        return net

    def create_policy_args(self, actor_size, actor_idx, critic_size, critic_idx):
        return dict(activation_fn=th.nn.ReLU,
                    net_arch=[dict(pi=self.create_net(actor_size, actor_idx),
                                   vf=self.create_net(critic_size, critic_idx))])
