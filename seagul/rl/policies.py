import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size , output_size, num_layers, layer_size, activation):
        super(MLP, self).__init__()

        self.activation = activation()

        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)])
        self.layers.extend([nn.Linear(layer_size, layer_size) for _ in range(num_layers )])
        self.output_layer = nn.Linear(layer_size, output_size)

    def forward(self, data):
        for layer in self.layers:
            data = self.activation(layer(data))

        return self.output_layer(data)


class Categorical_MLP(nn.Module):

    def __init__(self, input_size , output_size, num_layers, layer_size, activation):
        super(Categorical_MLP, self).__init__()

        self.activation = activation()

        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)])
        self.layers.extend([nn.Linear(layer_size, layer_size) for _ in range(num_layers )])
        self.output_layer = nn.Linear(layer_size, output_size)

        if output_size == 1:
            self.output_norm = nn.Sigmoid()
        else:
            self.output_norm = nn.Softmax(dim=-1)

    def forward(self, data):
        for layer in self.layers:
            data = self.activation(layer(data))

        return self.output_norm(self.output_layer(data))




if __name__ == '__main__':
    policy = MLP(input_size = 4, output_size = 1, num_layers = 3, layer_size = 12, activation=nn.ReLU)
    print(policy(torch.randn(1,4)))