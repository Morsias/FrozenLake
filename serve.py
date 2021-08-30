import gym
from starlette.requests import Request
import requests

import ray
import ray.rllib.agents.ppo as ppo
from ray import serve


@serve.deployment(route_prefix="/frozenlake-ppo")
class ServePPOModel:
    def __init__(self, checkpoint_path) -> None:
        self.trainer = ppo.PPOTrainer(
            config={
                "framework": "tf",
                # only 1 "local" worker with an env (not really used here).
                "num_workers": 0,
            },
            env="FrozenLake8x8-v1")
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]

        action = self.trainer.compute_single_action(obs, explore=False)
        return {"action": int(action)}



if __name__ == '__main__':
    # TODO comments
    CHECKPOINT_PATH = "models/PPO_FrozenLake8x8-v1_2021-08-29_12-44-17hwu4pe41/checkpoint_000210/checkpoint-210"
    serve.start()
    ServePPOModel.deploy(CHECKPOINT_PATH)

    env = gym.make("FrozenLake8x8-v1")
    observation = env.reset()
    for _ in range(50):

        print(f"-> Sending observation {observation}")
        resp = requests.get(
            "http://localhost:8000/frozenlake-ppo",
            json={"observation": observation})
        print(f"<- Received response {resp.json()}")
        observation, reward, done, info = env.step(resp.json()['action'])