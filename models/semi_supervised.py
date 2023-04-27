import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .fpn import FPN

class SemiSupervisedNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.ema_decay = 0.99

        self.b1 = FPN(pretrained="random", classes=1)
        self.b2 = FPN(pretrained="random", classes=1)
        
        for param in self.b2.parameters():
            param.detach_()
            
        for t_param, s_param in zip(self.b2.parameters(), self.b1.parameters()):
            t_param.data.copy_(s_param.data)
        
    def forward(self, x, update_w: bool = False, training: bool = True):
        
        if not self.training:
            pred = self.b1(x)
            return pred
          
        s_out = self.b1(x)
        
        with torch.no_grad():
            t_out = self.b2(x)
            
        if update_w:
            self._update_ema_variables(self.ema_decay) 
            
        return s_out, t_out
    
    def _update_ema_variables(self, ema_decay):
        for t_param, s_param in zip(self.b2.parameters(), self.b1.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

if __name__ == "__main__":
    m = SemiSupervisedNetwork()

    imgs = torch.rand((2, 3, 256, 256))
    masks = torch.rand((2, 1, 256, 256))
    uimgs = torch.rand((2, 3, 256, 256))

    sup_loss_fn = nn.BCEWithLogitsLoss()
    unsup_loss_fn = nn.MSELoss(reduction='mean')    

    spreds, tpreds = m(imgs, update_w = True)
    sunpreds, tunpreds = m(uimgs, update_w = False)

    s_pred = torch.cat([spreds, sunpreds], dim=0)
    t_pred = torch.cat([tpreds, tunpreds], dim=0)

    loss_unsup = unsup_loss_fn(
        F.sigmoid(s_pred).round(),
        F.sigmoid(t_pred).round().detach()
    )

    # supervised loss
    loss_sup = sup_loss_fn(spreds, masks)
    print(loss_sup, loss_unsup)
