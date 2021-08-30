import argparse
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
from datetime import datetime
import pandas as pd


def train_agent(iterations: int, save_results: bool, method: str):
    """
    This is the main training function for the FrozenLake8x8 environment. It trains a PPO or DQN agent and saves the
    checkpoints of the models every 50 iterations. Additionally, the mean reward per iteration is kept
    Args:
        iterations (int): Number of training iterations done by the agent. Each iteration number a larger number of
        timesteps. E.g. PPO runs 4000 steps per iteration
        save_results (bool): Whether the training results should be saved in a csv file for further analysis
        method (str): Rl methodology used. The two options currently are "PPO" and "DQN"

    Returns:

    """
    exp_time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    ray.init()
    if method == "PPO":
        config = ppo.DEFAULT_CONFIG.copy()
        # These create multiple parallel environments which massively speeds up training
        # the values when chosen after some experimentation in my own laptop
        config["num_workers"] = 3
        config['num_envs_per_worker'] = 8
        trainer = ppo.PPOTrainer(config=config, env="FrozenLake8x8-v1")
    elif method == "DQN":
        config = dqn.DEFAULT_CONFIG.copy()
        config["num_workers"] = 3
        config['num_envs_per_worker'] = 8
        config['timesteps_per_iteration'] = 4000
        trainer = dqn.DQNTrainer(config=config, env="FrozenLake8x8-v1")
    else:
        raise Exception("The method provided is not supported please choose PPO or DQN")




    mean_reward_results = []
    for i in range(iterations):
        # Perform one iteration of training the policy
        result = trainer.train()
        print("episode_reward_max: %s" % (result['episode_reward_max']))
        print("episode_reward_mean: %s" % (result['episode_reward_mean']))
        print("episodes_this_iter:%s" % (result['episodes_this_iter']))
        print("time_this_iter_s: %s" % (result['time_this_iter_s']))
        mean_reward_results.append(result['episode_reward_mean'])

        # Create a model checkpoint every 50 training iterations
        if i % 50 == 0:
            checkpoint = trainer.save(
                checkpoint_dir="models/%s_%s" % (method, exp_time))
            print("checkpoint saved at", checkpoint)


    if save_results:
        results_df = pd.DataFrame(mean_reward_results, columns=["MeanEpisodeReward"])
        results_df.to_csv("results/training_results_%s.csv" % exp_time)

    print("Total training time: %s secs" % result['time_total_s'])


parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    default="DQN",
    help="RL Methodology used. Can run with PPO and DQN")
parser.add_argument(
    "--iters",
    type=int,
    default=5,
    help="Number of iterations to train.")
parser.add_argument(
    "--save-results",
    type=bool,
    default=True,
    help="Save the episode reward")

if __name__ == '__main__':
    args = parser.parse_args()
    train_agent(args.iters, args.save_results, args.method)
    # ray.shutdown()
