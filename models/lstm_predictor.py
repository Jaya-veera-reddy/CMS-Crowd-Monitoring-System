import torch
import torch.nn as nn

class CrowdPredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=1,hidden_size=32,num_layers=2)
        self.fc = nn.Linear(32,1)

    def forward(self,x):

        out,_ = self.lstm(x)
        out = self.fc(out[-1])
        return out