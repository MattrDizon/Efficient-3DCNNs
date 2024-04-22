import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device("cuda:0") 

class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB, num_features):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(2*num_features, num_features)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)
        self.cuda()
        x = self.classifier(F.relu(x))
        return x