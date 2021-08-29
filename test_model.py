import ray
import ray.rllib.agents.ppo as ppo
import gym

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
agent = ppo.PPOTrainer(config=config, env="FrozenLake8x8-v1")

checkpoint_path = "models/PPO_FrozenLake8x8-v1_2021-08-29_12-44-17hwu4pe41/checkpoint_000210/checkpoint-210"
# checkpoint_path = "models/PPO_FrozenLake8x8-v1_2021-08-29_13-37-38u6chvsr1"
agent.restore(checkpoint_path)


env = gym.make("FrozenLake8x8-v1")
results = []
for i in range(1000):
    episode_reward = 0
    observation = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.compute_single_action(observation, explore=False)
        observation, reward, done, info = env.step(action)
        print(observation)
        episode_reward += reward
    results.append(episode_reward)

print(sum(results) / len(results))
print(max(results))