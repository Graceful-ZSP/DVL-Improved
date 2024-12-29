################## Imports ##################

import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityNet1(nn.Module):
    def __init__(self):
        super(VelocityNet1, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=6,
                      kernel_size=2, stride=1),
            nn.Tanh(),
        )
        self.ConvToFc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        )

        self.FC_output = nn.Sequential(
            nn.Linear(4, 2),
        )
        self.lstm = nn.LSTM(input_size=114, hidden_size=128, batch_first=False)

    def forward(self, x1, x2, y):
        x1 = self.conv_layer(x1)
        x2 = self.conv_layer(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x_lstm1 = x1.unsqueeze(0)
        x_lstm1, _ = self.lstm(x_lstm1)
        x_lstm1 = x_lstm1[-1]
        x_lstm2 = x2.unsqueeze(0)
        x_lstm2, _ = self.lstm(x_lstm2)
        x_lstm2 = x_lstm2[-1]
        x = torch.column_stack((x_lstm1, x_lstm2))
        x = F.dropout(x, p=0.2)
        x = self.ConvToFc(x)
        x = torch.column_stack((x, y))
        x = self.FC_output(x)
        return x

