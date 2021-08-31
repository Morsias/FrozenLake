import unittest
import ray
import gym

from inference import inference_model


class TestAgentInference(unittest.TestCase):

    def test_environment(self):
        # This function tests whether the environment runs as expected
        env = gym.make("FrozenLake8x8-v1")
        observation = env.reset()
        self.assertEqual(observation, 0)

        action = 0
        observation, reward, done, info = env.step(action)
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertAlmostEqual(info['prob'], 0.3333333333333333)

    def test_agent_inference(self):
        self.assertRaises(Exception, inference_model, "wrong_string", 5, False)
        ray.shutdown()
        model_path = "models/PPO_2021_08_31_09_37_06/checkpoint_000200/checkpoint-200"
        agent = inference_model(checkpoint_path=model_path, episodes=5, save_results=False, render=False)

        action = agent.compute_single_action(0, explore=False)
        self.assertEqual(action, 2)
