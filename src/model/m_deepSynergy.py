import torch.nn as nn
import torch


class DeepSynergyModel(nn.Module):
    def __init__(self, layers, input_size, input_dropout=0.2, dropout=0.5):
        super(DeepSynergyModel, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(layers)
        for i in range(self.num_layers):
            if i == 0:
                layer = nn.Linear(input_size, layers[i])
                nn.init.kaiming_normal_(layer.weight)
                self.layers.append(layer)
                self.layers.append(nn.Dropout(input_dropout))
            else:
                layer = nn.Linear(layers[i - 1], layers[i])
                nn.init.kaiming_normal_(layer.weight)
                self.layers.append(layer)
                if i < self.num_layers - 1:  # No dropout after the last
                    self.layers.append(nn.Dropout(dropout))

        self.activation = torch.nn.functional.relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                # Apply activation function except for the output layer
                if i < len(self.layers) - 2:  # Check if not last Linear layer
                    x = self.activation(x)
            else:
                x = layer(x)
        return x
