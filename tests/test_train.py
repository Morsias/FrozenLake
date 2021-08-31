import unittest

import ray
import gym

from train import train_agent

class TestAgentTraining(unittest.TestCase):


    def test_train_agent(self):

        self.assertRaises(Exception, train_agent, 10, True, "wrong_string")
        ray.shutdown()
        agent = train_agent(iterations=2, save_results=False, method="DQN")
