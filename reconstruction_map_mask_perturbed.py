import torch
from tqdm import tqdm
import math
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import sys
import os
import numpy as np
sys.path += ['../', '../../', os.path.dirname(__file__)]
from mri_forward_utils import center_crop


def inner_solver_with_mask(y, c_k, model, lmbd=1, H=None, Ht=None, op_norm=1, x_gt=None, toll=-1, prec=False, **kwargs):
    """solving the inverse problem with CRR-NNs using the adaptive gradient descent scheme"""

    max_iter = kwargs.get('max_iter', 1000)

    crop_img = kwargs.get('crop', False)
    
    pbar = tqdm(range(max_iter), dynamic_ncols=True)
    
    alpha = 1 / (op_norm ** 2)
    d_k =c_k
    x_old =c_k
    t_k = 5/3

    model.lmbda.data = (torch.ones(1, 1) * lmbd * alpha).to(y.device)

    if prec:
        n = 0
        tols = list()
        r = 3
    else:
        n = 50
        r = 3
        q = ((toll/r)/(toll*r)) ** (1/n)
        tols = [(toll*r) * q**(jj) for jj in range(n+1)]
    for _ in range(max_iter-n):
        tols.append(toll/r)

    for i in pbar:

        z = d_k - alpha * (Ht(H(d_k)-y))
        x = model.prox_denoise_with_mask(z, x_old, 500, tols[i])
        t_kp1 = (i + 6) / 3
        d_kp1 = x + (t_k - 1)/t_kp1 * (x - x_old)

        # relative change of norm for terminating
        res = torch.norm(x - x_old).double() / torch.norm(x_old).double()

        x_old = x.clone()
        t_k = t_kp1
        d_k = d_kp1.clone()
        
        if x_gt is not None:
            if crop_img:
                x_crop = center_crop(x, [320,320])
                psnr_ = psnr(x_crop, x_gt, data_range=1.0)
                ssim_ = ssim(x_crop, x_gt, data_range=1.0)
            else:
                psnr_ = psnr(x, x_gt, data_range=1.0)
                ssim_ = ssim(x, x_gt, data_range=1.0)
  
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            pbar.set_description(f"psnr: {psnr_:.2f} | res: {res:.2e}")
            psnr_ = None
            ssim_ = None

        if res < toll and (i > 10):
            break
                
    return(x, psnr_, ssim_, i)

def inner_solver_no_mask(y, model, lmbd=1, H=None, Ht=None, op_norm=1, x_gt=None, toll=-1, prec=False, **kwargs):

    max_iter = kwargs.get('max_iter', 1000)
    crop_img = kwargs.get('crop', False)
    
    pbar = tqdm(range(max_iter), dynamic_ncols=True)
    
    alpha = 1 / (op_norm ** 2)
    d_k = Ht(y)
    x_old = Ht(y)
    t_k = 5/3 

    model.lmbda.data = (torch.ones(1, 1) * lmbd * alpha).to(y.device)

    if prec:
        n = 0
        tols = list()
        r = 3
    else:
        n = 50
        r = 3
        q = ((toll/r)/(toll*r)) ** (1/n)
        tols = [(toll*r) * q**(jj) for jj in range(n+1)]
    for _ in range(max_iter-n):
        tols.append(toll/r)

    for i in pbar:
        
        z = d_k - alpha * (Ht(H(d_k)-y))
        x = model.prox_denoise_no_mask(z, x_old, 500, tols[i])
        t_kp1 = (i + 6) / 3
        d_kp1 = x + (t_k - 1)/t_kp1 * (x - x_old)

        # relative change of norm for terminating
        res = torch.norm(x - x_old).double() / torch.norm(x_old).double()

        x_old = x.clone()
        t_k = t_kp1
        d_k = d_kp1.clone()
        
        if x_gt is not None:
            if crop_img:
                x_crop = center_crop(x, [320,320])
                psnr_ = psnr(x_crop, x_gt, data_range=1.0)
                ssim_ = ssim(x_crop, x_gt, data_range=1.0)
            else:
                psnr_ = psnr(x, x_gt, data_range=1.0)
                ssim_ = ssim(x, x_gt, data_range=1.0)
  
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            pbar.set_description(f"psnr: {psnr_:.2f} | res: {res:.2e}")
            psnr_ = None
            ssim_ = None

        if res < toll and (i > 10):
            break
                
    return(x, psnr_, ssim_, i)




def center_crop(data, shape):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]



def prox_Recon(y, model, lmbd=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):

    with torch.no_grad():
        
        perturb = kwargs.get('perturb', False)
        seed = kwargs.get('perturb_seed', False)
        start_noise = kwargs.get('start_noise', False)


        s = 1e-3
        n = 5
        e = 1e-5
        q = (e/s) ** (1/n)
        tols = [s * q**(i) for i in range(n+1)]
        n_out = 10
        prec = False

        for _ in range(n_out-(n+1)):
            tols.append(e)

        (x, psnr_, ssim_, i) = inner_solver_no_mask(y, model, lmbd, H, Ht, op_norm, x_gt, tols[0], prec=prec, **kwargs)

        if perturb:
            torch.manual_seed(seed)
            x = x + torch.randn_like(x) * 15 / 255
            x_crop = center_crop(x, [320,320])
            psnr_ = psnr(x_crop, x_gt, data_range=1.0).item()

        if start_noise:
            torch.manual_seed(seed)
            x = torch.randn_like(x) 
            x_crop = center_crop(x, [320,320])
            psnr_ = psnr(x_crop, x_gt, data_range=1.0).item()

        print('perturbed PSNR (sol no mask, init mm):', psnr_)
        
        for it in range(n_out - 1):
            model.cal_mask(x)
            (x, psnr_, ssim_, i) = inner_solver_with_mask(y, x, model, lmbd, H, Ht, op_norm, x_gt, tols[it+1], prec=prec, **kwargs)
            

        print('final')

    return(x, psnr_, ssim_, i)