import ray 
from ray import tune 
from ray.tune.registry import register_env 

import argparse 
import json 

from ray.rllib.env.multi_agent_env import MultiAgentEnv  


class RayAviary(CustomizedWindEnv, MultiAgentEnv):
    def __init__(self, config={}):
        super().__init__(config)


    ray.init()
    config = {
        "env": Environment_Class,
        "num_workers":rllib_config['num_workers'],
        "num_gpus": rllib_config['num_gpus'],
        "env_config": env_config,
        "framework":"torch"
    }
    if multiagent_config:
        config["multiagent"]=multiagent_config
    
    analysis = tune.run(
        rllib_config['model'],
        config=config,
        stop = rllib_config['stop'],
        checkpoint_freq = rllib_config['checkpoint_freq'],
        checkpoint_at_end=True,
        local_dir = rllib_config['local_dir'],
        name = rllib_config['name'],
        restore= checkpoint
    )

    print("--------------------")
    print("Train is finished!")
    print("--------------------")
    
def test(Environment_Class, rllib_config, env_config, checkpoint, multiagent_config=None):
    ray.init()
    config = {
        "env": Environment_Class,
        "num_workers":0,
        "num_gpus": rllib_config['num_gpus'],
        "env_config": env_config,
        "framework":"torch"
    }
    if multiagent_config:
        config["multiagent"]=multiagent_config

    if rllib_config['model'] == "PPO":
        from ray.rllib.agents.ppo import PPOTrainer
        agent = PPOTrainer(config=config, env=Environment_Class)
    else:
        ValueError()
    
    agent.restore(checkpoint)
    env = Environment_Class(env_config)
    Reward = [] 
    for i in range(100):
        obs = env.reset()
        done = False
        Reward.append(0)
        while not done:
            alive_agents = [i for i in range(env_config['num_agents'])]
            actions =  {i:agent.compute_action(obs[i], policy_id=f"pol_{i}")  for i in alive_agents}
            obs, reward, done, info = env.step(actions)
            done = done["__all__"]
            Reward[-1] += sum(reward.values())/len(reward.values())
            print(i,  Reward[-1])

    print("--------------------")
    print("Test is finished!")
    print("--------------------")

def construct_multiagent_config():
    from gym.spaces import Box, Discrete
    multi_config = {
        "policies":{f"pol" : (None, Box(-np.inf, np.inf, shape=(3,)), Discrete(3) , {})}, # obs, act
        "policy_mapping_fn": lambda i : f"pol",
        "policies_to_train":[f"pol"],
        "observation_fn" : None # preprocess observation for each agent
    }
    return multi_config

def observation_fn(agent_obs):
    new_obs = {}
    for i in agent_obs.keys():
        new_obs[i] = agent_obs[i].flatten()
    return new_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument('--test', action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f :
        config = json.load(f)
        rllib_config = config['rllib_config']
        env_config = config['env_config']
        multi_agent_config  = construct_multiagent_config() #None  # set None if it is not multiagent setting

    Environment_Class = None 

    if args.test:
        test(Environment_Class, rllib_config, env_config, args.checkpoint, multi_agent_config)
    else:
        train(Environment_Class, rllib_config, env_config, args.checkpoint, multi_agent_config)
        
        
# config 
{
    "env_config":{
        "EPISODE_LEN_SEC" : 3
    },
    "rllib_config":{
        "model" : "PPO",
        "name" : "test",
        "num_workers" : 0,
        "num_gpus" : 0,
        "local_dir" : "./checkpoint",
        "checkpoint_freq" : 100,
        "framework" : "torch",
        "stop":{
            "training_iteration" : 300
        },
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.1,
            "epsilon_timesteps": 10000
        }
    }
}