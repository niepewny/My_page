import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRNNCell(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=3, kernel_size=5, depth=1, activation=F.relu):
        super().__init__()
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.out_channels = 1

        self.input_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=input_channels if i == 0 else hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ) for i in range(depth)
        ])

        self.hidden_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ) for i in range(depth)
        ])

        self.output_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels if i != depth-1 else self.out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ) for i in range(depth)
        ])

        self.bias = nn.Parameter(torch.randn(hidden_channels))

        self.activation = activation

        self.h_prev = None

    def initialize_hidden_state(self, batch_size, height, width, device):
        self.h_prev = torch.zeros(
            batch_size, self.hidden_channels, height, width, device=device
        )

    def forward(self, x, gen_output=False):
        if self.h_prev is None:
            batch_size, _, height, width = x.size()
            self.initialize_hidden_state(batch_size, height, width, x.device)

        for layer in self.input_layers:
            x = self.activation(layer(x))

        h_prev = self.h_prev
        for layer in self.hidden_layers:
            h_prev = self.activation(layer(h_prev))

        h_next = F.tanh(x + h_prev + self.bias.view(1, -1, 1, 1))

        self.h_prev = h_next

        # output = h_next
        # if gen_output:
        #     for i, layer in enumerate(self.output_layers):
        #         if i == len(self.output_layers) - 1:
        #             output = layer(output)
        #         else:
        #             output = self.activation(layer(output))

        return h_next
