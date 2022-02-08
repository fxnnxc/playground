# https://github.com/ray-project/ray/blob/7ff1cbbb12e5d7bc57d20bbdd90b7b951f6392e7/rllib/agents/dqn/dqn_torch_policy.py#L19

import gym 
import ray 

from custom_model import DeepingDQNTorchModel
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.policy.policy import Policy 
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise 
from ray.rllib.models.catalog import ModelCatalog

import numpy as np 
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override
from ray.rllib.policy.torch_policy import TorchPolicy

from ray.rllib.utils.torch_utils import (
    huber_loss,
    apply_grad_clipping,
    concat_multi_gpu_td_errors,
)

Q_SCOPE = "q_func"
Q_TARGET_SCOPE = "target_q_func"

# Importance sampling weights for prioritized replay
PRIO_WEIGHTS = "weights"


class QLoss:
    def __init__(
        self, 
        q_target_selected,
        q_logits_t_selected,
        q_tp1_best,
        q_props_tp1_best, 
        importance_weights,
        rewards,
        done_mask, 
        gamma=0.99,
        n_step=1,
        num_atoms=1, 
        v_min=-10.0,
        v_max=10.0):
        
        
        if num_atoms > 1:
            raise NotImplementedError("distributional RL")
    
        else:
            q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best 
            q_t_selected_target = rewards + gamma ** n_step * q_tp1_best_masked  # bellman equation

            # compute the error 
            self.td_error = q_target_selected - q_t_selected_target.detach()
            self.loss = torch.mean(
                importance_weights.float() * huber_loss(self.td_error)
            )
            self.stats = {
                "mean_q":torch.mean(q_target_selected),
                "min_q":torch.min(q_target_selected),
                "max_q":torch.max(q_target_selected),
            }

class ComputeTDErrorMixin:
    def __init__(self):
        def compute_td_error(
            obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights
        ):
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_t})
            input_dict[SampleBatch.ACTIONS] = act_t
            input_dict[SampleBatch.REWARDS] = rew_t
            input_dict[SampleBatch.NEXT_OBS] = obs_tp1
            input_dict[SampleBatch.DONES] = done_mask
            input_dict[PRIO_WEIGHTS] = importance_weights

            # Do forward pass on loss to update td error attribute
            build_q_losses(self, self.model, None, input_dict)

            return self.model.tower_stats["q_loss"].td_error

        self.compute_td_error = compute_td_error



class TargetNetworkMixin:
    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target()

    def update_target(self):
        state_dict = self.model.state_dict()
        for target in self.target_models.values():
            target.load_state_dict(state_dict)

    @override(TorchPolicy)
    def set_weights(self, weights):
        TorchPolicy.set_weights(self, weights)
        self.update_target()


def compute_q_values(policy, 
                     model, 
                     input_dict, 
                     state_batches=None, 
                     seq_lens=None, 
                     explore=None, 
                     is_training=False):
    
    model_out, state = model(input_dict, state_batches or [], seq_lens)

    config = policy.config
    if config['num_atoms']>1:
        pass 
    else:
        (action_scores, logits, probs_or_logits) = model.get_q_value_distributions(model_out)

    value = action_scores
    return value, logits, probs_or_logits, state


def grad_process_and_td_error_fn(
    policy: Policy, optimizer: "torch.optim.Optimizer", loss):
    return apply_grad_clipping(policy, optimizer, loss)


def extra_action_out_fn(
    policy: Policy, input_dict, state_batches, model, action_dist):
    return {"q_values": model.tower_stats["q_values"]}

def setup_early_mixins(
    policy, obs_space, action_space, config
    ):
    LearningRateSchedule.__init__(policy, config['lr'], config['lr_schedule'])

def before_loss_init(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config,
) -> None:
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)

def adam_optimizer(policy: Policy, config):

    # By this time, the models have been moved to the GPU - if any - and we
    # can define our optimizers using the correct CUDA variables.
    if not hasattr(policy, "q_func_vars"):
        policy.q_func_vars = policy.model.variables()

    return torch.optim.Adam(
        policy.q_func_vars, lr=policy.cur_lr, eps=config["adam_epsilon"]
    )



def build_q_stats(policy: Policy, batch):
    stats = {}
    for stats_key in policy.model_gpu_towers[0].tower_stats["q_loss"].stats.keys():
        stats[stats_key] = torch.mean(
            torch.stack(
                [
                    t.tower_stats["q_loss"].stats[stats_key].to(policy.device)
                    for t in policy.model_gpu_towers
                    if "q_loss" in t.tower_stats
                ]
            )
        )
    stats["cur_lr"] = policy.cur_lr
    return stats



def build_q_model_and_distribution(policy, obs_space, action_space, config):
    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=config["model"],
        model_interface=DeepingDQNTorchModel,
        name=Q_SCOPE
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=config["model"],
        model_interface=DeepingDQNTorchModel,
        name=Q_SCOPE
    )
    return model, TorchCategorical


def build_q_losses(policy, model, _, train_batch:SampleBatch):
    config = policy.config 

    # Q-network
    q_t, q_logits_t, q_probs_t, _ = compute_q_values(policy, model, 
                                    {"obs": train_batch[SampleBatch.CUR_OBS]}, explore=False, is_training=True,)
    # Target Q-network
    q_tp1, q_logits_tp1, q_probs_tp1, _ = compute_q_values(
        policy,
        policy.target_models[model],
        {"obs": train_batch[SampleBatch.NEXT_OBS]},
        explore=False,
        is_training=True,
    )

def get_distribution_inputs_and_class(
    policy: Policy,
    model,
    input_dict: SampleBatch,
    *,
    explore: bool = True,
    is_training: bool = False,
    **kwargs):
    q_vals = compute_q_values(
        policy, model, input_dict, explore=explore, is_training=is_training
    )
    q_vals = q_vals[0] if isinstance(q_vals, tuple) else q_vals

    model.tower_stats["q_values"] = q_vals

    return q_vals, TorchCategorical, []  # state-out

def postprocess_nstep_and_prio(
    policy: Policy, batch: SampleBatch, other_agent=None, episode=None
) -> SampleBatch:
    # N-step Q adjustments.
    if policy.config["n_step"] > 1:
        adjust_nstep(policy.config["n_step"], policy.config["gamma"], batch)

    # Create dummy prio-weights (1.0) in case we don't have any in
    # the batch.
    if PRIO_WEIGHTS not in batch:
        batch[PRIO_WEIGHTS] = np.ones_like(batch[SampleBatch.REWARDS])

    # Prioritize on the worker side.
    if batch.count > 0 and policy.config["worker_side_prioritization"]:
        td_errors = policy.compute_td_error(
            batch[SampleBatch.OBS],
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.REWARDS],
            batch[SampleBatch.NEXT_OBS],
            batch[SampleBatch.DONES],
            batch[PRIO_WEIGHTS],
        )
        new_priorities = (
            np.abs(convert_to_numpy(td_errors))
            + policy.config["prioritized_replay_eps"]
        )
        batch[PRIO_WEIGHTS] = new_priorities

    return batch

DeepingDQNTorchPolicy = build_policy_class(
    name="DQNTorchPolicy",
    framework="torch",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ],
)