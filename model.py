import gym
import torch as th

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

class RLAlgorithm:
    def __init__(self):
        # Create environment
        self.env = gym.make('LunarLander-v2')
        self.hyperparameters = Hyperparameters()
        self.model = None

    def create_model(self):
        raise NotImplementedError('Please implement this method')

    def train(self, total_timesteps, model_name):
        # Choose hyperparameters
        self.choose_hyperparameters()
        # Create the model
        self.create_model()
        # Train the model
        self.model.learn(total_timesteps=total_timesteps)
        # Save the model
        self.model.save(model_name)

    def evaluate(self, model_name, n_eval_episodes):
        self.load_model_from_file(model_name)
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=n_eval_episodes)

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

    def choose_hyperparameters(self):
        for learning_idx in range(4):
            for actor_idx in range(3):
                for critic_idx in range(3):
                    learning_rate = self.hyperparameters.create_learning_rate(learning_idx)
                    policy_kwargs = self.hyperparameters.create_policy_args(2, actor_idx, 2, critic_idx)
                    return learning_rate, policy_kwargs


class PPOAlgorithm(RLAlgorithm):
    def create_model(self):
        # Create the agent
        self.model_name = "ppo_lunar"
        self.model = PPO("MlpPolicy", self.env, learning_rate=self.learning_rate, policy_kwargs=self.policy_kwargs, verbose=1)

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


learning rate: 0.0001 -> 0.00008 -> 0.00006 -> 0.00004
redes:
    actor: 2 camadas: [32, 32], [64, 64], [128, 128]
    critic: 2 camadas: [32, 32], [64, 64], [128, 128]
