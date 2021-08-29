import argparse
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import time
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    default="PPO",
    help="RL Methodology used. Can run with PPO and DQN")
parser.add_argument(
    "--iters",
    type=int,
    default=200,
    help="Number of iterations to train.")
parser.add_argument(
    "--save-results",
    type=bool,
    default=True,
    help="Save the episode reward")

if __name__ == '__main__':
    args = parser.parse_args()
    start_time = time.time()

    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 3
    config['num_envs_per_worker'] = 8

    trainer = ppo.PPOTrainer(config=config, env="FrozenLake8x8-v1")
    mean_reward_results = []
    for i in range(args.iters):
        # Perform one iteration of training the policy
        result = trainer.train()
        print("episode_reward_max:" + result['episode_reward_max'])
        print("episode_reward_mean:" + result['episode_reward_mean'])
        print("episodes_this_iter:" + result['episodes_this_iter'])
        print("time_this_iter_s:" + result['time_this_iter_s'])
        mean_reward_results.append(result['episode_reward_mean'])

        # Create a model checkpoint every 50 training iterations
        if i % 50 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

        print("--- The training took %s seconds ---" % (time.time() - start_time))


    if args.save_results:
        results_df = pd.DataFrame(mean_reward_results, columns=["MeanEpisodeReward"])
        results_df.to_csv("results/training_results.csv")
    ray.shutdown()
