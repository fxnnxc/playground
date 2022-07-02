# https://theaisummer.com/weights-and-biases-tutorial/
# https://docs.wandb.ai/ref/python/init

import wandb 
import numpy as np 
wandb.init(project="my-test-project")

wandb.config.learning_rate =0.1

# wandb.watch(net, criterion, log="all")
# wandb.watch(net, criterion, log="all")

import torch 
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linaer1 = nn.Linear(64, 32)
        self.linaer2 = nn.Linear(32, 64)
    def forward(self, x):
        x = self.linaer1(x)
        x = nn.functional.relu(x) 
        x = self.linaer2(x)
        return x 




net = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=wandb.config.learning_rate)

wandb.watch(net, criterion, log="all", log_freq=10)

for epoch in range(10):
    running_loss = 0.0
    for i in range(1000):
        inputs = torch.rand(32, 1,64) 
        labels = inputs

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            wandb.log({'epoch': epoch+1, 'loss': running_loss/20})
            running_loss = 0.0

print('Finished Training')