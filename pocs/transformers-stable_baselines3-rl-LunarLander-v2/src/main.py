import gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2")
model = PPO(policy = "MlpPolicy",
            env = env,
            n_steps = 1024,
            batch_size = 64,
            n_epochs = 10,
            verbose = 1)

model.learn(total_timesteps = 1000)
model_name = "PPO-LunarLander-v2"
model.save(model_name)
print(f"Model {model_name} saved. Done!")