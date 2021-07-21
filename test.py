import gym
import torch as th

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO


# Create environment
env = gym.make('LunarLander-v2')

# Custom actor (pi) and value function (vf) networks
# of two layers of size 32 each with Relu activation function
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[32, 32], vf=[32, 32])])
# Create the agent
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# Train the agent
model.learn(total_timesteps=int(1e5))
# Save the agent
model.save("ppo_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO.load("ppo_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("mean_reward:", mean_reward, "std_reward:", std_reward)

# Enjoy trained agent
obs = env.reset()
for i in range(int(2e5)):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
