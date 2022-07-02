# https://theaisummer.com/weights-and-biases-tutorial/
# https://docs.wandb.ai/ref/python/init

import wandb 
import numpy as np 
wandb.init(project="my-test-project", entity='bumjin')

wandb.config.learning_rate =0.1

for i in range(100):
    wandb.log({"epcoh":i, "loss" : np.random.random()/(1+i)}, step=i)
    if i%10 == 0:
        wandb.log({"residual":i}, step=i)


# wandb.watch(net, criterion, log="all")
# wandb.watch(net, criterion, log="all")

