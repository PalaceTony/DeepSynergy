import torch.nn as nn


class DeepSynergyModel(nn.Module):
    def __init__(self, input_size, layers, input_dropout, dropout, act_func):
        super(DeepSynergyModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, layers[i]))
                self.layers.append(nn.Dropout(input_dropout))
            else:
                self.layers.append(nn.Linear(layers[i - 1], layers[i]))
                if i < len(layers) - 1:  # No dropout after the last layer
                    self.layers.append(nn.Dropout(dropout))
        self.activation = act_func

    def forward(self, x):
        for layer in self.layers:
            x = (
                self.activation(layer(x))
                if not isinstance(layer, nn.Dropout)
                else layer(x)
            )
        return x
