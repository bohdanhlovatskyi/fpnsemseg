import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class D(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, p, z):
        # return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

        z = z.detach()

        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        
        return -(p * z).sum(dim=1).mean()
    
class Projector(nn.Module):
    
    def __init__(self,
                 in_dim: int,
                 h_dim: int = 2048,
                 out_dim: int = 2048
    ) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(*[
            nn.Linear(in_dim, h_dim), 
            nn.BatchNorm1d(h_dim), 
            nn.ReLU(inplace=True),
        ])
        
        self.layer2 = nn.Sequential(*[
            nn.Linear(h_dim, h_dim), 
            nn.BatchNorm1d(h_dim), 
            nn.ReLU(inplace=True),
        ])
        
        self.layer3 = nn.Sequential(*[
            nn.Linear(h_dim, out_dim), 
            nn.BatchNorm1d(out_dim), 
        ])
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x
    
class Predictor(nn.Module):
    
    def __init__(self,
                 in_dim: int = 2048,
                 h_dim: int = 2048,
                 out_dim: int = 2048
    ) -> None:
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(h_dim, out_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        
        return x
    
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        encoder = torchvision.models.resnet34(weights=None)
        
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.projector = Projector(encoder.fc.in_features)
        self.predictor = Predictor()
        
        self.D = D()
        
    def forward(self, x1, x2):
        e1, e2 = self.encoder(x1).squeeze(), self.encoder(x2).squeeze()
        
        z1 = self.projector(e1)
        z2 = self.projector(e2)
        
        p1, p2 = self.predictor(z1), self.predictor(z2)
            
        return 0.5 * self.D(p1, z2) + 0.5 * self.D(p2, z1)

if __name__ == "__main__":
    m = Model()

    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    loss = m(x1, x2)
    print(loss)
    loss.backward()
