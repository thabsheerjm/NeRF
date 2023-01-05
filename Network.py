import torch 
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self,f = 6):
        super().__init__()
        self.layer1 = nn.Linear(3+3*2*f,256)  # f is number of embeds
        self.layer2 = nn.Linear(256,256,nn.ReLU())
        self.layer3 = nn.Linear(256,1)  # sigma out
        self.layer4 = nn.Linear(256+3+3*2*f,128)
        self.layer5 = nn.Linear(128,3)  # 3 channels out


    def forward(self, x, d):
        x =F.relu(self.layer1(x))
        x =F.relu(self.layer2(x))
        sigma = self.layer3(x)
        x = torch.concat([x,d],dim = -1)
        x = F.relu(self.layer4(x))
        rgb = self.layer5(x)
        x=torch.concat([rgb, sigma], dim =1)

        return x

