import torch.nn as nn
import torch


class ThreeMLPdrugSynergyModel(nn.Module):
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
        super(ThreeMLPdrugSynergyModel, self).__init__()

        # Define the first sub-network
        self.dsn1 = nn.Sequential()
        for i, layer_size in enumerate(dsn1_layers):
            if i == 0:
                self.dsn1.add_module(f"dsn1_fc{i}", nn.Linear(input_shape1, layer_size))
            else:
                self.dsn1.add_module(
                    f"dsn1_fc{i}", nn.Linear(dsn1_layers[i - 1], layer_size)
                )
            self.dsn1.add_module(f"relu{i}", nn.ReLU())
            self.dsn1.add_module(
                f"dropout{i}", nn.Dropout(p=0.2)
            )  # Adjust dropout rate as needed

        # Define the second sub-network
        self.dsn2 = nn.Sequential()
        for i, layer_size in enumerate(dsn2_layers):
            if i == 0:
                self.dsn2.add_module(f"dsn2_fc{i}", nn.Linear(input_shape2, layer_size))
            else:
                self.dsn2.add_module(
                    f"dsn2_fc{i}", nn.Linear(dsn2_layers[i - 1], layer_size)
                )
            self.dsn2.add_module(f"relu{i}", nn.ReLU())
            self.dsn2.add_module(
                f"dropout{i}", nn.Dropout(p=0.2)
            )  # Adjust dropout rate as needed

        # Cell Line sub-network
        self.cln = nn.Sequential()
        for i, layer_size in enumerate(cln_layers):
            if i == 0:
                self.cln.add_module(
                    f"cln_fc{i}", nn.Linear(input_shape_cln, layer_size)
                )
            else:
                self.cln.add_module(
                    f"cln_fc{i}", nn.Linear(cln_layers[i - 1], layer_size)
                )
            self.cln.add_module(f"cln_relu{i}", nn.ReLU())
            self.cln.add_module(
                f"cln_dropout{i}", nn.Dropout(p=0.2)
            )  # Adjust dropout rate as needed

        # Output Network
        self.spn = nn.Sequential()
        # Adjust combined_input_size to include output of the cln network
        combined_input_size = dsn1_layers[-1] + dsn2_layers[-1] + cln_layers[-1]
        for i, layer_size in enumerate(spn_layers):
            if i == 0:
                self.spn.add_module(
                    "spn_fc{0}".format(i), nn.Linear(combined_input_size, layer_size)
                )
            else:
                self.spn.add_module(
                    "spn_fc{0}".format(i), nn.Linear(spn_layers[i - 1], layer_size)
                )
            self.spn.add_module("spn_relu{0}".format(i), nn.ReLU())
            if i < len(spn_layers) - 1:  # Add dropout to all layers except the last one
                self.spn.add_module("spn_dropout{0}".format(i), nn.Dropout(p=0.2))

        # Output layer
        self.output = nn.Linear(spn_layers[-1], 1)

    def forward(self, drug1, drug2, cell_line):
        x1 = self.dsn1(drug1)
        x2 = self.dsn2(drug2)
        x3 = self.cln(cell_line)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.spn(x)
        return self.output(x)
