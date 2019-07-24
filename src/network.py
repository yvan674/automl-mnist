"""Network.

This is the network that will be used. It's a simple convolutional network that
outputs a single image class classification

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, num_conv_layers: int, kernel_size: int, activation: str,
                 fc_units: int, per_layer_dropout: float,
                 fc_dropout: float) -> None:
        """Initializes the network.

        Args:
            num_conv_layers: Number of convolutional layers that the network
                will use.
            kernel_size: Kernel size of the convolutional layers.
            activation: Activation type of each layer. Possible options are:
                - 'relu'
                - 'leaky relu'
                - 'sigmoid'
            per_layer_dropout: Dropout for each layer.
            fc_dropout: Final dropout layer value.
        """
        super().__init__()
        # Figure out which activation to use
        if activation == "relu":
            activation_func = nn.ReLU()

        elif activation == "leaky relu":
            activation_func = nn.LeakyReLU()

        elif activation == 'sigmoid':
            activation_func = nn.Sigmoid()

        else:
            raise ValueError("Activation type given is not one of the possible "
                             "options")

        # Create convolutional layers
        self.convolutionals = []
        channels = 1
        for i in range(num_conv_layers):
            out_channels = 2 ** (3 + i)
            print("Producing Conv2d with {} in and {} out".format(channels, out_channels))
            layer = nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size),
                activation_func,
                nn.Dropout2d(per_layer_dropout)
            )
            self.convolutionals.append(layer)
            channels = out_channels

        # Calculate number of features
        self.num_flat_features = self.hacky_features_calculation()

        # create FC layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_flat_features, fc_units),
            activation_func)
        self.fc2 = nn.Sequential(
            nn.Linear(fc_units, 10),
            activation_func)

        # Save final dropout value
        self.fc_dropout = fc_dropout


    def forward(self, input):
        x = input

        # Iterate through convolutions
        for convolution in self.convolutionals:
            x = convolution(x)

        x = x.view(-1, self.num_flat_features)

        x = self.fc1(x)
        x = F.dropout(x, self.fc_dropout)
        x = self.fc2(x)

        return x

    def hacky_features_calculation(self) -> int:
        """Calculates number of features after convolutions.

        Calculates this by passing a random tensor through the network and
        calculating the number of features present.
        """
        x = torch.randint(0, 255, [1, 28, 28], dtype=torch.float)
        x = x.unsqueeze(0)

        for convolution in self.convolutionals:
            x = convolution(x)

        size = x.shape[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
