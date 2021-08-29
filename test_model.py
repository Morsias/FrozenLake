import argparse
import ray
import ray.rllib.agents.ppo as ppo
import gym
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="models/PPO_FrozenLake8x8-v1_2021-08-29_12-44-17hwu4pe41/checkpoint_000210/checkpoint-210",
    help="RL model that will be restored")
parser.add_argument(
    "--iters",
    type=int,
    default=1000,
    help="Number of iterations of inference.")
parser.add_argument(
    "--render",
    type=bool,
    default=False,
    help="Render environment during inference")
parser.add_argument(
    "--save-results",
    type=bool,
    default=True,
    help="Save the episode reward")


if __name__ == '__main__':
    args = parser.parse_args()

    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    agent = ppo.PPOTrainer(config=config, env="FrozenLake8x8-v1")

    agent.restore(args.checkpoint_path)


    env = gym.make("FrozenLake8x8-v1")
    results = []
    for i in range(args.iters):

        episode_reward = 0
        observation = env.reset()
        done = False
        while not done:
            if args.render:
                env.render()
            action = agent.compute_single_action(observation, explore=False)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
        results.append(episode_reward)


    if args.save_results:
        results_df = pd.DataFrame(results, columns=["EpisodeReward"])
        results_df.to_csv("results/inference_results.csv")
