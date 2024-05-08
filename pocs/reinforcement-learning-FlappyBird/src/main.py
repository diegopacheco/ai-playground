import gymnasium as gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import flappy_bird_env  # noqa


env = gym.make("FlappyBird-v0", render_mode="rgb_array")
observation, info = env.reset(seed=42)
recorder = VideoRecorder(env, path='fb.mp4')

try:
    for i in range(5000):
        print(f"step {i}")
        action = env.action_space.sample() # samples random action from action sample space
        # the agent takes the action
        observation, reward, terminated, trunc, info = env.step(action)  # Corrected this line

        env.render()
        recorder.capture_frame()
        print(f"observation: {observation}")

        # if the agent reaches terminal state, we reset the environment
        if terminated:
            print("Environment is reset")
            observation = env.reset()
finally:
    recorder.close()
    env.close()