import torch
import torch.nn as nn


class ThreeMLPdrugSynergyModel(nn.Module):
    def __init__(
        self,
        dsn1_layers,
        dsn2_layers,
        cln_layers,
        spn_layers,
        drugA_input_shape,
        drugB_input_shape,
        cell_line_input_shape,
        input_dropout=0.2,
        hidden_dropout=0.5,
    ):
        super(ThreeMLPdrugSynergyModel, self).__init__()

        # Drug A sub-network
        self.dsn1 = nn.ModuleList()
        input_drop_out_flag = True
        prev_size = drugA_input_shape
        for layer_size in dsn1_layers[:-1]:
            self.dsn1.append(nn.Linear(prev_size, layer_size))
            self.dsn1.append(nn.ReLU())
            drop_out = input_dropout if input_drop_out_flag else hidden_dropout
            input_drop_out_flag = False
            self.dsn1.append(nn.Dropout(drop_out))
            prev_size = layer_size
        self.dsn1.append(nn.Linear(prev_size, dsn1_layers[-1]))

        # Drug B sub-network
        self.dsn2 = nn.ModuleList()
        input_drop_out_flag = True
        prev_size = drugB_input_shape
        for layer_size in dsn2_layers[:-1]:
            self.dsn2.append(nn.Linear(prev_size, layer_size))
            self.dsn2.append(nn.ReLU())
            drop_out = input_dropout if input_drop_out_flag else hidden_dropout
            input_drop_out_flag = False
            self.dsn1.append(nn.Dropout(drop_out))
            prev_size = layer_size
        self.dsn2.append(nn.Linear(prev_size, dsn2_layers[-1]))

        # Cell Line sub-network
        self.cln = nn.ModuleList()
        input_drop_out_flag = True
        prev_size = cell_line_input_shape
        for layer_size in cln_layers[:-1]:
            self.cln.append(nn.Linear(prev_size, layer_size))
            self.cln.append(nn.ReLU())
            drop_out = input_dropout if input_drop_out_flag else hidden_dropout
            input_drop_out_flag = False
            self.dsn1.append(nn.Dropout(drop_out))
            prev_size = layer_size
        self.cln.append(nn.Linear(prev_size, cln_layers[-1]))

        # Combined Network
        self.spn = nn.ModuleList()
        input_drop_out_flag = True
        combined_input_size = dsn1_layers[-1] + dsn2_layers[-1] + cln_layers[-1]
        prev_size = combined_input_size
        for layer_size in spn_layers[:-1]:  # Not the last layer
            self.spn.append(nn.Linear(prev_size, layer_size))
            self.spn.append(nn.ReLU())
            self.spn.append(nn.Dropout(p=0.2))
            prev_size = layer_size
        self.spn.append(nn.Linear(prev_size, spn_layers[-1]))

        # Output
        self.output = nn.Linear(spn_layers[-1], 1)

    def forward(self, drugA, drugB, cell_line):
        for layer in self.dsn1:
            drugA = layer(drugA)
        for layer in self.dsn2:
            drugB = layer(drugB)
        for layer in self.cln:
            cell_line = layer(cell_line)

        x = torch.cat((drugA, drugB, cell_line), dim=1)

        for layer in self.spn:
            x = layer(x)

        x = self.output(x)
        return x
