import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device("cuda:0") 

class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB, num_features):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Sequential(
            nn.Linear(2 * 30, 30),
            nn.ReLU(),
            nn.Linear(30, 30)
        )
        self.to(device)
        self.cuda()
        

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x