import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
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
        self.simulation_timesteps = int(1e4)
        self.n_eval_episodes = 20
        self.models = []
        self.lr_size = 4
        self.a_size = 3
        self.c_size = 3

    def grid_search(self):
        hyp_id = 0
        for learning_idx in range(self.lr_size):
            for actor_idx in range(self.a_size):
                for critic_idx in range(self.c_size):
                    hyp_id += 1
                    dictionary, model_name = self.train(learning_idx, actor_idx, critic_idx, hyp_id)
                    self.evaluate(model_name, dictionary)

    def train(self, learning_idx, actor_idx, critic_idx, hyp_id):
        # Choose hyperparameters
        learning_rate, policy_kwargs, policy_info = self.choose_params(learning_idx, actor_idx, critic_idx)
        # Create the model
        model_name = self.create_model(learning_rate, policy_kwargs, hyp_id)
        # Train the model
        self.model.learn(total_timesteps=self.train_timesteps)
        # Save the model
        self.model.save(model_name)
        return self.create_model_info_dict(model_name, learning_rate, policy_info), model_name

    def choose_params(self, learning_idx, actor_idx, critic_idx):
        learning_rate = self.params.create_learning_rate(learning_idx)
        policy_kwargs, policy_info = self.params.create_policy_kwargs(2, actor_idx, 2, critic_idx)
        return learning_rate, policy_kwargs, policy_info

    def create_model(self, learning_rate, policy_kwargs, hyp_id):
        raise NotImplementedError('Please implement this method')

    @staticmethod
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    def create_model_info_dict(self, model_name, learning_rate, policy_info):
        return {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "policy_kwargs": policy_info
        }

    def evaluate(self, model_name, dictionary):
        model = self.load_model_from_file(model_name)
        dictionary['mean_reward'], dictionary['std_reward'] = evaluate_policy(model, model.get_env(), n_eval_episodes=self.n_eval_episodes)
        self.saveJSON(model_name, dictionary)
        self.models.append((dictionary['mean_reward'], model_name))

    def load_model_from_file(self, model_name):
        raise NotImplementedError('Please implement this method')

    def saveJSON(self, file_name, dictionary):
        print(dict)
        with open("models/" + file_name + ".json", "w") as outfile:
            json.dump(dictionary, outfile, indent=4)

    def create_results_file(self):
        self.models.sort(reverse=True)
        print(self.models)
        f = open("results.txt", "w")
        for model in self.models:
            line = "Model: " + model[1] + " Mean reward: " + str(model[0]) + "\n"
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

    def record_video(self, video_name, video_length, model_name):
        record_env = DummyVecEnv([lambda: gym.make('LunarLander-v2')])
        obs = record_env.reset()
        record_env = VecVideoRecorder(record_env, 'recordings/',
                               record_video_trigger=lambda x: x == 0, video_length=video_length,
                               name_prefix=video_name)
        record_env.reset()
        model = self.load_model_from_file(model_name)
        for _ in range(video_length + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = record_env.step(action)
        env.close()
