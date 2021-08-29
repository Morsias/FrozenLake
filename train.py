# from ray import tune
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import gym

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 3
config['num_envs_per_worker'] = 8


trainer = ppo.PPOTrainer(config=config, env="FrozenLake8x8-v1")

for i in range(200):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print("episode_reward_max:" + result['episode_reward_max'])
   print("episode_reward_mean:" + result['episode_reward_mean'])
   print("episodes_this_iter:" + result['episodes_this_iter'])
   print("time_this_iter_s:" + result['time_this_iter_s'])

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)

# tune.run(ppo.PPOTrainer, config={"env": "FrozenLake8x8-v0"})