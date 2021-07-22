from rl import RLAlgorithm
from stable_baselines3 import PPO

class PPOAlgorithm(RLAlgorithm):
    def create_model(self, learning_rate, policy_kwargs, hyp_id):
        self.algo_name = "ppo"
        # Create the agent
        self.model = PPO("MlpPolicy", self.env, learning_rate=learning_rate, policy_kwargs=policy_kwargs, verbose=1)
        return self.algo_name + "_" + str(hyp_id)

    def load_model_from_file(self, model_name):
        self.model = PPO.load(model_name, env=self.env)
