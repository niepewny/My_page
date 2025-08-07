import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=3, kernel_size=5, depth=1, activation=F.relu):
        super().__init__()
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.out_channels = hidden_channels
        self.combined_channels = 4*hidden_channels

        # idea of combining conv layers taken from: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
        self.combined_conv_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=hidden_channels + input_channels if i == 0 else self.combined_channels,
                out_channels=self.combined_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ) for i in range(depth)
        ])

        self.activation = activation

        self.H = None
        self.C = None

    def initialize_hidden_state(self, batch_size, height, width, device):
        self.H = torch.zeros(
            batch_size, self.hidden_channels, height, width, device=device
        )
        self.C = torch.zeros(
            batch_size, self.hidden_channels, height, width, device=device
        )

    # todo: gen_output is only a placeholder for now
    def forward(self, x, gen_output=False):
        if self.H is None:
            batch_size, _, height, width = x.size()
            self.initialize_hidden_state(batch_size, height, width, x.device)

        combined = torch.cat([x, self.H], dim=1)        

        for i in range(self.depth):
            if i != self.depth - 1:
                combined = self.activation(self.combined_conv_layers[i](combined))
            else:
                combined = self.combined_conv_layers[i](combined)

        combined_F_I_O = torch.sigmoid(combined[:, :3*self.hidden_channels])
        F, I, O = torch.split(combined_F_I_O, split_size_or_sections=self.hidden_channels, dim=1)
        CC = torch.tanh(combined[:, 3*self.hidden_channels:])

        II = I*CC

        self.C = self.C * F + II
        self.H = O * torch.tanh(self.C)

        return self.H