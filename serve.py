import gym
from starlette.requests import Request
import requests

import ray
import ray.rllib.agents.ppo as ppo
from ray import serve


@serve.deployment(route_prefix="/frozenlake-ppo")
class ServePPOModel:
    def __init__(self, checkpoint_path) -> None:
        # Initialize trainer and restore model
        self.trainer = ppo.PPOTrainer(
            config={
                "framework": "tf",
                # only 1 "local" worker with an env (not really used here).
                "num_workers": 0,
            },
            env="FrozenLake8x8-v1")
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request: Request):
        # Get request compute optimal action and return in json form
        json_input = await request.json()
        obs = json_input["observation"]

        action = self.trainer.compute_single_action(obs, explore=False)
        return {"action": int(action)}



if __name__ == '__main__':
    # Setup server
    CHECKPOINT_PATH = "models/PPO_2021_08_31_09_37_06/checkpoint_000200/checkpoint-200"
    serve.start()
    ServePPOModel.deploy(CHECKPOINT_PATH)

    # Setup env for inference
    env = gym.make("FrozenLake8x8-v1")
    observation = env.reset()
    for _ in range(50):
        # Send observation to API
        print(f"-> Sending observation {observation}")
        resp = requests.get(
            "http://localhost:8000/frozenlake-ppo",
            json={"observation": observation})
        print(f"<- Received response {resp.json()}")
        observation, reward, done, info = env.step(resp.json()['action'])