import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="rgb_array")
model = PPO.load("PPO-LunarLander-v2",env=env)
recorder = VideoRecorder(env, path='lunar.mp4')
env = model.get_env()

obs = env.reset()
for i in range(1000):
    # getting action predictions from the trained agent
    action, _states = model.predict(obs, deterministic=True)

    # taking the predicted action in the environment to observe next state and rewards
    obs, rewards, dones, info = env.step(action)
    print(f"obs = {obs}, rewards = {rewards}, dones = {dones}, info = {info}")

    env.render()
    recorder.capture_frame()

recorder.close()
env.close()