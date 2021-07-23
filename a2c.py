from rl import RLAlgorithm
from stable_baselines3 import A2C


class A2CAlgorithm(RLAlgorithm):
    def create_model(self, learning_rate, gamma, policy_kwargs, hyp_id):
        self.algo_name = "a2c"
        # Create the agent
        model = A2C("MlpPolicy", self.env, learning_rate=self.linear_schedule(learning_rate), gamma=gamma,
                    policy_kwargs=policy_kwargs, verbose=1)
        model_name = self.algo_name + "_" + str(hyp_id)
        return model, model_name

    def load_model_from_file(self, model_name):
        return A2C.load(model_name, env=self.env)
