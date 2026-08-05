import torch
import numpy as np
import torch.nn as nn

X = np.load("outputs/X.npy")
y = np.load("outputs/y.npy")

X = torch.tensor(X).float().unsqueeze(-1)
y = torch.tensor(y).float().unsqueeze(-1)

class CrowdPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(1,32,batch_first=True)
        self.fc = nn.Linear(32,1)

    def forward(self,x):

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        out = self.fc(out)

        return out


model = CrowdPredictor()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(200):

    pred = model(X)

    loss = criterion(pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("Epoch",epoch,"Loss",loss.item())

torch.save(model.state_dict(),"checkpoints/lstm_model.pth")

print("LSTM training complete")