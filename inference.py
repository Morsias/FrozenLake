import argparse
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import gym
import pandas as pd
from datetime import datetime
import logging


def inference_model(checkpoint_path: str, episodes: int, save_results: bool = True, render: bool = False):
    """
    This is the main testing function for the FrozenLake8x8 environment. It loads a PPO or DQN agent and runs an
    inference process. The results can be saved in .csv.
    Args:
        render: (bool): Whether the environment should be rendered or not
        checkpoint_path (str): The path of the saved RL agent that the user wants to restore. The path must end on
        a folder that includes a .tune_metadata file
        episodes (int): Number of inference episodes done by the agent
        save_results (bool): Whether the training results should be saved in a csv file for further analysis

    Returns:

    """
    exp_time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    ray.init()
    if "PPO" in checkpoint_path:
        method = "PPO"
        config = ppo.DEFAULT_CONFIG.copy()
        agent = ppo.PPOTrainer(config=config, env="FrozenLake8x8-v1")
    elif "DQN" in checkpoint_path:
        method = "DQN"
        config = dqn.DEFAULT_CONFIG.copy()
        agent = dqn.DQNTrainer(config=config, env="FrozenLake8x8-v1")
    else:
        raise Exception("The given checkpoint is not currently supported. Please load a DQN or PPO checkpoint")

    agent.restore(checkpoint_path)

    env = gym.make("FrozenLake8x8-v1")
    results = []
    for i in range(episodes):

        episode_reward = 0
        observation = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = agent.compute_single_action(observation, explore=False)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
        logging.info('Episode %s reward: %s' % (i, episode_reward))
        results.append(episode_reward)

    if save_results:
        results_df = pd.DataFrame(results, columns=["EpisodeReward"])
        result_path = "results/inference_results_%s_%s.csv" % (method, exp_time)
        results_df.to_csv(result_path)
        logging.info('Results were saved at: %s' % result_path)

    return agent


parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="models/PPO_2021_08_31_09_37_06/checkpoint_000200/checkpoint-200",
    help="RL model that will be restored")
# For DQN use path:
# models/DQN_2021_08_31_09_29_14/checkpoint_000200/checkpoint-200
parser.add_argument(
    "--episodes",
    type=int,
    default=1000,
    help="Number of episodes of inference")
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
    agent = inference_model(args.checkpoint_path, args.episodes, args.save_results, args.render)
    ray.shutdown()
