

import logging 
"""
Based on 
https://github.com/ray-project/ray/blob/7ff1cbbb12e5d7bc57d20bbdd90b7b951f6392e7/rllib/agents/dqn/dqn_torch_model.py#L7
"""

import torch 
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from deeping.module.nn import CNNForNatureDQN

class DeepingDQNTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name:str, * , dueling=False, dueling_activation='relu'):
        nn.Module.__init__(self)
        super(DeepingDQNTorchModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        if model_config['type'] == "CNNForNatureDQN":
            self.advantage_module = CNNForNatureDQN(action_space.n)
        else:
            raise ValueError("Not Impelemted Model Type")

    def get_q_value_distributions(self, model_out):
        self.model_out = model_out
        print(model_out.size())
        logging.log(model_out.size())
        action_scores = self.advantage_module(model_out)
        logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
        return action_scores, logits, logits