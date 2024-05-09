### Result
* Reinforcement learning algorithm: PPO
* Environment: LunarLander-v2

##### Training
```
./run.sh
```
Training output:
```
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 90.6     |
|    ep_rew_mean     | -128     |
| time/              |          |
|    fps             | 1196     |
|    iterations      | 1        |
|    time_elapsed    | 0        |
|    total_timesteps | 1024     |
---------------------------------
Model PPO-LunarLander-v2 saved. Done!
```
##### Eval
```
./eval.sh
```
Eval output:
```
mean_reward=-548.29 +/- 86.65819220670235
```
##### Inference
```
./inference.sh
```
Inference output:
```

```