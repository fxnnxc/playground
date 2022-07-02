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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.ImageNet(root="/data3/bumjin_data/ILSVRC2012_val", split='val', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

columns=['image','label']
data = []

for i, batch in enumerate(testloader, 0):
    inputs, labels = batch[0], batch[1]
    for j, image in enumerate(inputs,0):
        data.append([wandb.Image(image),classes[labels[1].item()]])
    break

table= wandb.Table(data=data, columns=columns)
wandb.log({"cifar10_images": table})
