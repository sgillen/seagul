from seagul.nn import MLP, fit_model
import torch
import torch.nn as nn
import time

net1 = nn.Sequential(
    nn.Linear(4,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32,1)
)

net2 = MLP(4,1,num_layers=2, layer_size=32)

X = torch.rand(40960,4)
Y = torch.rand(40960,1)

start = time.time()
fit_model(net1, X,Y,num_epochs=10)
print(time.time() - start)

start = time.time()
fit_model(net2, X,Y,num_epochs=10)
print(time.time() - start)