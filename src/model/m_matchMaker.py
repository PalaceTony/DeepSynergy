import torch
import torch.nn as nn


class MatchMaker(nn.Module):
    def __init__(
        self,
        dsn1_layers,
        dsn2_layers,
        cln_layers,
        spn_layers,
        input_shape1,
        input_shape2,
        input_shape_cln,
    ):
        super(MatchMaker, self).__init__()

        # Drug A sub-network
        self.dsn1 = nn.ModuleList()
        prev_size = input_shape1
        for layer_size in dsn1_layers:
            self.dsn1.append(nn.Linear(prev_size, layer_size))
            self.dsn1.append(nn.ReLU())
            self.dsn1.append(nn.Dropout(p=0.2))
            prev_size = layer_size

        # Drug B sub-network
        self.dsn2 = nn.ModuleList()
        prev_size = input_shape2
        for layer_size in dsn2_layers:
            self.dsn2.append(nn.Linear(prev_size, layer_size))
            self.dsn2.append(nn.ReLU())
            self.dsn2.append(nn.Dropout(p=0.2))
            prev_size = layer_size

        # Cell Line sub-network
        self.cln = nn.ModuleList()
        prev_size = input_shape_cln
        for layer_size in cln_layers:
            self.cln.append(nn.Linear(prev_size, layer_size))
            self.cln.append(nn.ReLU())
            self.cln.append(nn.Dropout(p=0.2))
            prev_size = layer_size

        # Combined Network
        self.spn = nn.ModuleList()
        combined_input_size = dsn1_layers[-1] + dsn2_layers[-1] + cln_layers[-1]
        for layer_size in spn_layers[:-1]:  # Not the last layer
            self.spn.append(nn.Linear(combined_input_size, layer_size))
            self.spn.append(nn.ReLU())
            self.spn.append(nn.Dropout(p=0.2))
            prev_size = layer_size
        self.spn.append(nn.Linear(prev_size, spn_layers[-1]))

        # Output
        self.output = nn.Linear(spn_layers[-1], 1)

    def forward(self, drug1, drug2, cell_line):
        for layer in self.dsn1:
            drug1 = layer(drug1)
        for layer in self.dsn2:
            drug2 = layer(drug2)
        for layer in self.cln:
            cell_line = layer(cell_line)

        x = torch.cat((drug1, drug2, cell_line), dim=1)

        for layer in self.spn:
            x = layer(x)

        x = self.output(x)
        return x
