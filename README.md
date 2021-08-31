# FrozenLake8x8-v1 Solution
The goal of the project is to solve the _FrozenLake8x8-v1_ environment. In order to achieve that 2 RL methodologies are tested namely:
- DQN
- PPO

The training code is located in train.py, while the inference code is located in inference.py.
Additionally, a serve.py file was created which showcases the easy integration of RLlib models to an API endpoint

### Setup Instructions

The packages must be installed in the following order:
1. ```pip install tensorflow```
2. ```pip install 'ray[rllib]'```
3. ```pip install 'ray[serve]'```
4. ```pip install jupyter```
5. ```pip install seaborn```
6. ```pip install starlette```


### Executing the code

- Running train.py </br>
``` python train.py --method "PPO" --iters 200 --save-results```

- Running inference.py for PPO model</br>
``` python inference.py --checkpoint_path "models/PPO_2021_08_31_09_37_06/checkpoint_000200/checkpoint-200" --episodes 10 --render --save-results```
- Running inference.py for DQN model </br>
``` python inference.py --checkpoint_path "models/DQN_2021_08_31_09_29_14/checkpoint_000200/checkpoint-200" --episodes 10 --render --save-results```

### Results Presentation
A presentation of the results can be found in results_presentation.ipynb 