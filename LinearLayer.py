import torch
from torch import nn 


def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-8)
    normalized_tensor = tensor / norm
    return normalized_tensor


class LinearLayer(nn.Module):
    def __init__(self, n, kernerl_size, conv_pad, n_filters, n_layer, groups=1) -> None:
        super().__init__()

        self.n = n
        self.kernel_size = kernerl_size
        self.conv_pad = conv_pad
        self.n_filters = n_filters
        self.n_layer = n_layer
        
        

        self.W1 = nn.Conv2d(self.n, self.n_filters, kernel_size=self.kernel_size, padding=self.conv_pad, bias=False, groups=groups)
        self.W1s = nn.ModuleList([nn.Conv2d(self.n_filters, self.n_filters, kernel_size=self.kernel_size, padding=self.conv_pad, bias=False, groups=groups) for _ in range(self.n_layer - 1)])

        self.W1T = nn.ConvTranspose2d(self.n_filters, self.n, kernel_size=self.kernel_size, padding=self.conv_pad, bias=False, groups=groups)
        self.W1sT = nn.ModuleList([nn.ConvTranspose2d(self.n_filters, self.n_filters, kernel_size=self.kernel_size, padding=self.conv_pad, bias=False, groups=groups) for _ in range(self.n_layer - 1)])

        self.W1T.weight.requires_grad_(False)
        for i in range(self.n_layer-1):
            self.W1sT[i].weight.requires_grad_(False)

        return
    
    def cal_lip(self, eigenimage, n_it=4):
        with torch.no_grad():
            for _ in range(n_it):
                v = normalize(self.Wt(eigenimage))
                eigenimage = normalize(self.W(v))
                v = v.clone()
        
        ct = torch.sum(eigenimage * self.W(v))**2

        return ct, eigenimage

    def W(self, x): 
        
        lx = self.W1(x)

        for i in range(self.n_layer-1):
            lx = self.W1s[i](lx)

        return lx
    
    def Wt(self, y):
    
        self.W1T.weight = self.W1.weight
        for i, _ in enumerate(self.W1sT): 
            self.W1sT[i].weight = self.W1s[i].weight

        if self.n_layer > 1: 
            lt = self.W1sT[self.n_layer-2](y) 
            for i in range(self.n_layer-2): 
                lt = self.W1sT[self.n_layer-i-3](lt)
            lt = self.W1T(lt)
        else: 
            lt = self.W1T(y) 

        return lt 