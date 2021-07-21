import gym
import torch as th

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

class RLAlgorithm:
    def __init__(self):
        raise NotImplementedError('Please implement this method')

    def create_model(self):
        raise NotImplementedError('Please implement this method')

    def train(self, total_timesteps):
        # Create the model
        self.create_model()
        # Train the model
        self.model.learn(total_timesteps=total_timesteps)
        # Save the model
        self.model.save(self.model_name)

    def evaluate(self, n_eval_episodes):
        self.load_model()
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=n_eval_episodes)
        print("mean_reward:", mean_reward)
        print("std_reward:", std_reward)

    def load_model(self):
        raise NotImplementedError('Please implement this method')

    def simulate(self, timesteps):
        # Enjoy trained agent
        obs = self.env.reset()
        for i in range(timesteps):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, info = self.env.step(action)
            self.env.render()


class PPO(RLAlgorithm):
    def __init__(self):
        # Create environment
        self.env = gym.make('LunarLander-v2')

    def load_hyperparameters(self):


    def create_model(self):
        # Create the agent
        self.model_name = "ppo_lunar"
        self.model = PPO("MlpPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)

    def load_model(self):
        self.model = PPO.load(self.model_name, env=self.env)

class Hyperparameters:

