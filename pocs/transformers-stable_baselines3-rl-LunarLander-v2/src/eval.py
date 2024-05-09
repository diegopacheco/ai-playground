import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2")
# Loading the saved model
model = PPO.load("PPO-LunarLander-v2",env=env)

# Initializating the evaluation environment
eval_env = gym.make("LunarLander-v2")

# Running the trained agent on eval_env for 10 time steps and getting the mean reward
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes = 10,
                                          deterministic=True)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")