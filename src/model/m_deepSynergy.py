import torch
import torch.nn as nn


class DeepSynergyModel(nn.Module):
    def __init__(self, layers, input_size, input_dropout=0.2, dropout=0.5):
        super(DeepSynergyModel, self).__init__()
        self.network = nn.ModuleList()

        first = True
        prev_size = input_size

        for layer_size in layers[:-1]:
            self.network.append(nn.Linear(prev_size, layer_size))
            self.network.append(nn.ReLU())
            if first:
                self.network.append(nn.Dropout(input_dropout))
                first = False
            else:
                self.network.append(nn.Dropout(dropout))
            prev_size = layer_size

        self.network.append(nn.Linear(prev_size, layers[-1]))

        # Kaiming Normal
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x
