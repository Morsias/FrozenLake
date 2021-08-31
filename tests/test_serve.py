import unittest
import requests
from ray import serve


from serve import ServePPOModel


class TestServing(unittest.TestCase):

    def test_serving(self):

        checkpoint_path = "models/PPO_2021_08_31_09_37_06/checkpoint_000200/checkpoint-200"
        serve.start()
        ServePPOModel.deploy(checkpoint_path)

        resp = requests.get(
            "http://localhost:8000/frozenlake-ppo",
            json={"observation": 0})
        self.assertEqual(resp.json()['action'], 2)