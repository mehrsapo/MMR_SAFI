import torch
from torch import nn 
import numpy as np
from spline_module import LinearSpline
import torch.nn.functional as F
from torch import autograd

from LinearLayer import * 

class RNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__() 

        self.kernel_size = config['kernel_size']
        self.conv_pad = config['kernel_size'] // 2
        self.n_filters = config['n_filter']
        self.n_layer = config['n_layer']
        self.n = 1  

        self.alpha = torch.ones(1, 1) * 1e-5
        self.lmbda = torch.nn.Parameter(torch.ones(1, 1) * 1e-4)

        self.W1 = LinearLayer(1, self.kernel_size, self.conv_pad, self.n_filters, self.n_layer)

        self.eigenimage =  normalize(torch.nn.Parameter(torch.randn(128, self.n_filters, 40, 40))).detach()

        self.phigrad_spline = LinearSpline(num_activations=self.n_filters, num_knots=21, x_min=0, x_max=1, init='ccv',
                    slope_max=0, slope_min=None, antisymmetric=False, clamp=False, oneatzero=True) 
        
        self.W2 = LinearLayer(self.n_filters, self.kernel_size, self.conv_pad, self.n_filters, self.n_layer, self.n_filters)
        
        self.cs = torch.nn.Parameter(torch.ones(1, self.n_filters, 1, 1))

        return
    
    def phigrad(self, x):
        return torch.clip(self.phigrad_spline(torch.abs(self.cs) * x), 0, 1)

    def cal_mask(self, y): 
        self.mask = self.W2.Wt(self.phigrad(self.W2.W(torch.abs(self.W1.W(y)))))

    def L(self, x, state='mask'):
        lx = self.W1.W(x)

        if state == 'mask':
            lx = lx * self.mask

        return lx
    

    def Lt(self, y, state='mask'):

        if state  == 'mask':
            y = y * self.mask

        ly = self.W1.Wt(y)
        return ly 

    def f_maj(self, u_k, y, state='mask'): 
        u_k = torch.clip(u_k + self.alpha * self.L(torch.clip(y - self.Lt(u_k, state), 0, 1), state), -self.lmbda , self.lmbda)   
        return u_k
    


    def prox_denoise_no_mask(self, y, c_k, n_iter, tol, check_tol=True):

        u_k = self.L(c_k, state='nomask')
        v_k = self.L(c_k, state='nomask')
        t_k = 5/3

        c_k_old = torch.clip(y - self.Lt(u_k, state='nomask'), 0, 1)  

        for iters in range(n_iter):
            u_kp1 = self.f_maj(v_k, y, state='nomask')
            t_kp1 = (iters + 6) / 3
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)     
            
            if (not self.training) and check_tol :
                c_k_new = torch.clip(y - self.Lt(u_kp1, state='nomask'), 0, 1)   
                rel_err = torch.norm(c_k_new - c_k_old) / torch.norm(c_k_old)  

                if rel_err < tol and (iters > 5):
                    break

                c_k_old = c_k_new

            u_k = u_kp1
            t_k = t_kp1
            v_k = v_kp1

        c_k = torch.clip(y - self.Lt(u_k, state='nomask'), 0, 1)  
        return c_k
    

    def prox_denoise_with_mask(self, y, c_k, n_iter, tol, check_tol=True):
        u_k = self.L(c_k)
        v_k = self.L(c_k)
        t_k = 5/3

        c_k_old = torch.clip(y - self.Lt(u_k), 0, 1)  

        for iters in range(n_iter):
            u_kp1 = self.f_maj(v_k, y)
            t_kp1 = (iters + 6) / 3
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)     
            
            if (not self.training) and check_tol:
                c_k_new = torch.clip(y - self.Lt(u_kp1), 0, 1)   
                rel_err = torch.norm(c_k_new - c_k_old) / torch.norm(c_k_old)  
                if rel_err < tol and (iters > 5):
                    break

                c_k_old = c_k_new

            u_k = u_kp1
            t_k = t_kp1
            v_k = v_kp1

        c_k = torch.clip(y - self.Lt(u_k), 0, 1)   
        return c_k
    

    def prox_denoise_with_mask(self, y, c_k, n_iter, tol, check_tol=True):
        u_k = self.L(c_k)
        v_k = self.L(c_k)
        t_k = 1 

        c_k_old = torch.clip(y - self.Lt(u_k), 0, 1)  
        loss_old = ((y - c_k_old)**2).sum() / 2 + self.lmbda * self.L(c_k_old).abs().sum()

        for iters in range(n_iter):
            u_kp1 = self.f_maj(v_k, y)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)     
            
            if (not self.training) and check_tol:
                c_k_new = torch.clip(y - self.Lt(u_kp1), 0, 1)   
                loss_new = ((y - c_k_new)**2).sum() / 2 + self.lmbda * self.L(c_k_new).abs().sum()
                
                rel_err = torch.abs(loss_old - loss_new) / loss_old  
                if rel_err < tol:
                    break

                loss_old = loss_new

            u_k = u_kp1
            t_k = t_kp1
            v_k = v_kp1

        c_k = torch.clip(y - self.Lt(u_k), 0, 1)   
        return c_k


    def solve_majorize_minimize(self, y):
        
        if self.training:
            n_out = torch.randint(4, 7, (1, 1))
            n_in = torch.randint(10, 13, (1, 1)) 
        else:
            n_out = 5 
            n_in = 12  
            
        lip, self.eigenimage = self.W1.cal_lip(self.eigenimage)
        self.alpha = 1 / lip

        u_k = self.L(y, state='nomask')
        v_k = self.L(y, state='nomask')
        t_k = 1 

        for _ in range(n_in):
            u_kp1 = self.f_maj(v_k, y, state='nomask')
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)     

            u_k = u_kp1
            t_k = t_kp1
            v_k = v_kp1

        
        c_k = torch.clip(y - self.Lt(u_k, state='nomask'), 0, 1)   
           
        for _ in range(n_out - 1):   
            self.cal_mask(c_k)
            t_k = 1
                    
            for _ in range(n_in):
                u_kp1 = self.f_maj(v_k, y)
                t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
                v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)     

                u_k = u_kp1
                t_k = t_kp1
                v_k = v_kp1
                
            c_k = torch.clip(y - self.Lt(u_k), 0, 1) 
        
        return c_k

    def sumtoone(self, X):
        X = torch.abs(X)
        Y = X / torch.sum(X, dim=(2,3), keepdim=True)
        return Y
    
    def zeromean(self, X):
        Y = X - torch.mean(X, dim=(2,3), keepdim=True)
        return Y

    def forward(self, y):
        
        self.lmbda.data = self.lmbda.data.clamp(1e-12, float('Inf'))

        self.W2.W1.weight.data = self.sumtoone(self.W2.W1.weight)
        for i, _ in enumerate(self.W2.W1s): 
            self.W2.W1s[i].weight.data = self.sumtoone(self.W2.W1s[i].weight)


        self.W1.W1.weight.data = self.zeromean(self.W1.W1.weight)
        for i, _ in enumerate(self.W1.W1s): 
            self.W1.W1s[i].weight.data = self.zeromean(self.W1.W1s[i].weight)

        x = self.solve_majorize_minimize(y)

        return x