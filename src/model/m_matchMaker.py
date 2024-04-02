import torch
import torch.nn as nn


class MatchMakerModel(nn.Module):
    def __init__(
        self,
        dsn1_layers,
        dsn2_layers,
        spn_layers,
        input_shape1,
        input_shape2,
        input_dropout=0.2,
        dropout=0.5,
    ):
        super(MatchMakerModel, self).__init__()

        input_drop_out_flag = True
        # Drug A sub-network
        self.dsn1 = nn.ModuleList()
        prev_size = input_shape1
        for layer_size in dsn1_layers[:-1]:
            self.dsn1.append(nn.Linear(prev_size, layer_size))
            self.dsn1.append(nn.ReLU())
            drop_out = input_dropout if input_drop_out_flag else dropout
            input_drop_out_flag = False
            self.dsn1.append(nn.Dropout(drop_out))
            prev_size = layer_size
        self.dsn1.append(nn.Linear(prev_size, dsn1_layers[-1]))

        input_drop_out_flag = True
        # Drug B sub-network
        self.dsn2 = nn.ModuleList()
        prev_size = input_shape2
        for layer_size in dsn2_layers[:-1]:
            self.dsn2.append(nn.Linear(prev_size, layer_size))
            self.dsn2.append(nn.ReLU())
            drop_out = input_dropout if input_drop_out_flag else dropout
            input_drop_out_flag = False
            self.dsn1.append(nn.Dropout(drop_out))
            prev_size = layer_size
        self.dsn2.append(nn.Linear(prev_size, dsn2_layers[-1]))

        # Combined Network
        self.spn = nn.ModuleList()
        combined_input_size = dsn1_layers[-1] + dsn2_layers[-1]
        for layer_size in spn_layers[:-1]:
            self.spn.append(nn.Linear(combined_input_size, layer_size))
            self.spn.append(nn.ReLU())
            self.spn.append(nn.Dropout(input_dropout))
            combined_input_size = layer_size
        self.spn.append(nn.Linear(combined_input_size, spn_layers[-1]))

        # Output layer
        self.output = nn.Linear(spn_layers[-1], 1)

    def forward(self, drugA_cell, drugB_cell):
        for layer in self.dsn1:
            drugA_cell = layer(drugA_cell)
        for layer in self.dsn2:
            drugB_cell = layer(drugB_cell)
        x = torch.cat((drugA_cell, drugB_cell), dim=1)
        for layer in self.spn:
            x = layer(x)
        x = self.output(x)
        return x
