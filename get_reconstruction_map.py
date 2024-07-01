import sys
import argparse
import torch
import os
import json

sys.path += ['../', '../../', os.path.dirname(__file__)]




def get_reconstruction_map(method, modality, device="cuda:0", **opts):
    if modality == 'ct':
        from ct.utils_ct.ct_forward_utils import get_operators as get_operators_ct
        from ct.utils_ct.ct_forward_utils import get_op_norm as get_op_norm_ct

        H, fbp, Ht = get_operators_ct(device=device)
        op_norm = get_op_norm_ct(H, Ht, device=device)

    elif modality == 'mri':
        from mri.utils_mri.mri_forward_utils import get_operators as get_operators_mri
        from mri.utils_mri.mri_forward_utils import get_op_norm as get_op_norm_mri

    tol = opts.get('tol', 1e-5)
    if method == 'crr':
        from utils_inverse_problems.reconstruction_map_crr import AdaGD_Recon, AGD_Recon
        from models.utils import load_model
        
        model_name = opts.get('model_name', None)

        if model_name is None:
            raise ValueError("Please provide a model_name for the crr model. It is the name of the folder corrsponding to the trained model.")
        
        model = load_model(model_name, epoch=10, device=device)
        model.eval()
        model.prune(prune_filters=True, collapse_filters=False, change_splines_to_clip=False)

        if modality == 'ct':
            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    algo_name = opts.get('algo_name', 'adagd')
                    if algo_name == 'adagd':
                        x, psnr_, ssim_, n_iter = AdaGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    elif algo_name == 'fista':
                        x, psnr_, ssim_, n_iter = AGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    else:
                        raise NotImplementedError
                return(x, psnr_, ssim_, n_iter)
        elif modality == 'mri':
            def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    H, Ht = get_operators_mri(mask, smap, device=device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = AGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_zf, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError


    elif method == 'prox_drunet':
        
        from utils_inverse_problems.reconstruction_map_prox_drunet import run_drs
        import torch
        sys.path += ['../../others/prox_drunet']
        sys.path += ['/home/pourya/all_methods/others/prox_drunet/PnP_restoration']
        import get_denoiser

        model_name = 'prox_drunet' #opts.get('model_name', None)
       

        if model_name is None:
            raise ValueError("Please provide a model_name for the mask model. It is the name of the folder corrsponding to the trained model.")
        

        if modality == 'ct':
            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    algo_name = opts.get('algo_name', 'adagd')
                    if algo_name == 'adagd':
                        x, psnr_, ssim_, n_iter = AdaGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    elif algo_name == 'fista':
                        x, psnr_, ssim_, n_iter = AGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    else:
                        raise NotImplementedError
                return(x, psnr_, ssim_, n_iter)
            
        elif modality == 'mri':
            def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    H, Ht, prox_op = get_operators_mri(mask, smap, device=device)
                    import get_denoiser
                    model = get_denoiser.get_denoiser(p1, device=y.device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = run_drs(y, model, p2=p2, H=H, Ht=Ht, prox_op=prox_op, x_gt=x_gt, x_init=x_zf, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError





    elif method == 'mask_rfdn':
        from utils_inverse_problems.reconstruction_map_mask_rfdn import prox_Recon
        sys.path.append('/home/pourya/Sebastian_Unet/')
        sys.path.append('/home/pourya/Sebastian_Unet/models')
        sys.path.append('/home/pourya/Sebastian_Unet/SwinIR')
        import utils as utils
        import json
        import copy
        from RFDN import RFDN
        import torch
        from dataloader.BSD500 import BSD500
        from models import deep_equilibrium
        #from swinir import SwinIR
        import numpy as np
        


        model_name = 'mask_RFDN_try2' #opts.get('model_name', None)
        fname = '/home/pourya/Sebastian_Unet/WCRR-CNN/'
        model = utils.load_model(fname, epoch=6000, device=device)
        model.activation_cvx.hyper_param_to_device()
        model.activation_ccv.hyper_param_to_device()
        model.update_integrated_params()
        model.eval()
        model.to(device)

        model_fix = utils.load_model(fname, epoch=6000, device=device)
        model_fix.to(device)
        model_fix.activation_cvx.hyper_param_to_device()
        model_fix.activation_ccv.hyper_param_to_device()
        model_fix.update_integrated_params()
        model_fix.eval()

        # high accuracy computation of \|W\|
        print(" **** Updating the Lipschitz constant **** ")
        sn_pm = model.conv_layer.spectral_norm(mode="power_method", n_steps=100)

        # load deafualts
        config_path = '/home/pourya/Sebastian_Unet/WCRR-CNN/config.json'
        config = json.load(open(config_path))

        my_nn = RFDN(in_nc=1, out_nc=80, upscale=1, nf = 40)
        my_nn.to(device)
        my_nn.load_state_dict(torch.load('/home/pourya/Sebastian_Unet/model_paras_denoised4_multi_deninp_small.pt'))

        '''config_path = '/home/pourya/Sebastian_Unet/SwinIR/config_mask.json'
        config = json.load(open(config_path))

        my_nn  = SwinIR(in_chans=1,out_chans=config['networks']['out_c'],
                        upscale=1,window_size=8,img_range=1., depths=[6, 6, 6, 6],
                        embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                        upsampler='pixelshuffledirect').to(device)


        weights = torch.load(f'/home/pourya/Sebastian_Unet/SwinIR/checkpoint_best_val.pth')
        my_nn.load_state_dict(weights['state_dict_mask'])
        model.load_state_dict(weights['state_dict'])'''
        my_nn.eval()

        #model.load_state_dict(torch.load('/home/pourya/Sebastian_Unet/updated_reg_new.pt'))

        if model_name is None:
            raise ValueError("Please provide a model_name for the mask model. It is the name of the folder corrsponding to the trained model.")
        
        

        #model.prune(prune_filters=True, collapse_filters=False, change_splines_to_clip=False)

        if modality == 'ct':
            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    algo_name = opts.get('algo_name', 'adagd')
                    if algo_name == 'adagd':
                        x, psnr_, ssim_, n_iter = AdaGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    elif algo_name == 'fista':
                        x, psnr_, ssim_, n_iter = AGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    else:
                        raise NotImplementedError
                return(x, psnr_, ssim_, n_iter)
        elif modality == 'mri':
            def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    H, Ht, _ = get_operators_mri(mask, smap, device=device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = prox_Recon(y, model, model_fix, my_nn, p1=p1, p2=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_zf, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError
        
    elif method == 'mask_relax':
        from utils_inverse_problems.reconstruction_map_mask import prox_Recon
        sys.path.append('/home/pourya/iterative_masks_final/')
        from R_network_relax import RNet 
        from models.utils import load_model
        import json, torch
        
        
        model_name = 'mask_relax' #opts.get('model_name', None)

        if model_name is None:
            raise ValueError("Please provide a model_name for the mask model. It is the name of the folder corrsponding to the trained model.")
        
        exp = "/home/pourya/iterative_masks_final/exps/sigma15/64_2_7_1e-3_relaxed"
        #exp = "exps/sigma_25_64_4_3_deq_shared/AdpReg_ccvcvx"
        path_ckp = exp + "/checkpoints/checkpoint_best_epoch.pth"
        path_config = exp + "/config.json"
        #device = 'cuda:3'
        config = json.load(open(path_config))
        ckp = torch.load(path_ckp, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

        model = RNet(config['model_params'])
        model.to(device)
        model.load_state_dict(ckp['state_dict'])
        model.eval()

        '''pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('parameters:', pytorch_total_params)'''

        model.W1.W1.weight.data = model.zeromean(model.W1.W1.weight)
        for i, _ in enumerate(model.W1.W1s): 
            model.W1.W1s[i].weight.data = model.zeromean(model.W1.W1s[i].weight)
                #model.prune(prune_filters=True, collapse_filters=False, change_splines_to_clip=False)

        model.eigenimage = model.eigenimage.to(device)
        lip, model.eigenimage = model.W1.cal_lip(model.eigenimage, 200)
        model.alpha = 1 / lip

        model.eigenimage = model.eigenimage.to(device)
        if modality == 'ct':
            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    algo_name = opts.get('algo_name', 'adagd')
                    if algo_name == 'adagd':
                        x, psnr_, ssim_, n_iter = AdaGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    elif algo_name == 'fista':
                        x, psnr_, ssim_, n_iter = AGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    else:
                        raise NotImplementedError
                return(x, psnr_, ssim_, n_iter)
        elif modality == 'mri':
            def reconstruction_map(y, mask, smap, p1, x_gt=None, x_init=None):
                with torch.no_grad():
                    H, Ht, _ = get_operators_mri(mask, smap, device=device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = prox_Recon(y, model, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, x_init=x_zf, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError
        
    
    elif method == 'mask_mm':
        from utils_inverse_problems.reconstruction_map_mask import prox_Recon
        sys.path.append('/home/pourya/iterative_masks_final/')
        from R_network_mm import RNet 
        from models.utils import load_model
        import json, torch
        
        model_name = 'mask_mm' #opts.get('model_name', None)

        if model_name is None:
            raise ValueError("Please provide a model_name for the mask model. It is the name of the folder corrsponding to the trained model.")
        
        exp = "/home/pourya/iterative_masks_final/exps/sigma15/64_2_7_1e-3_mm"
        #exp = "exps/sigma_25_64_4_3_deq_shared/AdpReg_ccvcvx"
        path_ckp = exp + "/checkpoints/checkpoint_best_epoch.pth"
        path_config = exp + "/config.json"
        #device = 'cuda:0'
        config = json.load(open(path_config))
        ckp = torch.load(path_ckp, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

        model = RNet(config['model_params'])
        model.to(device)
        model.load_state_dict(ckp['state_dict'])
        model.eval()


        model.W2.W1.weight.data = model.sumtoone(model.W2.W1.weight)
        for i, _ in enumerate(model.W2.W1s): 
            model.W2.W1s[i].weight.data = model.sumtoone(model.W2.W1s[i].weight)


        model.W1.W1.weight.data = model.zeromean(model.W1.W1.weight)
        for i, _ in enumerate(model.W1.W1s): 
            model.W1.W1s[i].weight.data = model.zeromean(model.W1.W1s[i].weight)


        model.eigenimage = model.eigenimage.to(device)
        lip, model.eigenimage = model.W1.cal_lip(model.eigenimage, 200)
        model.alpha = 1 / lip

        if modality == 'ct':
            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    algo_name = opts.get('algo_name', 'adagd')
                    if algo_name == 'adagd':
                        x, psnr_, ssim_, n_iter = AdaGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    elif algo_name == 'fista':
                        x, psnr_, ssim_, n_iter = AGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                    else:
                        raise NotImplementedError
                return(x, psnr_, ssim_, n_iter)
        elif modality == 'mri':
            def reconstruction_map(y, mask, smap, p1, x_gt=None, x_init=None):
                with torch.no_grad():
                    H, Ht, _ = get_operators_mri(mask, smap, device=device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = prox_Recon(y, model, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, x_init=x_zf, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError
    elif method == 'wcrr':
        from utils_inverse_problems.reconstruction_map_wcrr import SAGD_Recon
        sys.path += ['../../others/wcrr/model_wcrr/']
        from utils import load_model as load_model_wcrr
        import torch
        
        
        model_name = opts.get('model_name', 'WCRR-CNN')

        if model_name is None:
            raise ValueError("Please provide a model_name for the wcrr model. It is the name of the folder corrsponding to the trained model.")
        
        model = load_model_wcrr(model_name, device=device)
        model.eval()

        '''pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('parameters:', pytorch_total_params)'''

        sn_pm = model.conv_layer.spectral_norm(mode="power_method", n_steps=1000)


        if modality == 'ct':

            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    x, psnr_, ssim_, n_iter = SAGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        elif modality == 'mri':
            def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                with torch.no_grad():
                    H, Ht, _ = get_operators_mri(mask, smap, device=device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = SAGD_Recon(y, model, lmbd=p1, mu=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_zf, op_norm=op_norm, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError
        
    elif method == 'tv':
        if modality == 'ct':
            from utils_inverse_problems.reconstruction_map_tv import TV_Recon
            def reconstruction_map(y, p1, x_gt=None, x_init=None, **kwargs):
                with torch.no_grad():
                    x, psnr_, ssim_, n_iter = TV_Recon(y, alpha=1/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, **opts)
                return(x, psnr_, ssim_, n_iter)
        elif modality == 'mri':
            from utils_inverse_problems.reconstruction_map_tv import TV_Recon
            def reconstruction_map(y, mask, smap, p1, x_gt=None, x_init=None, **kwargs):
                with torch.no_grad():
                    H, Ht = get_operators_mri(mask, smap, device=device)
                    op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                    x_zf = Ht(y)
                    x, psnr_, ssim_, n_iter = TV_Recon(y, alpha=1/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, x_init=x_zf, **opts)
                return(x, psnr_, ssim_, n_iter)
        else:
            raise NotImplementedError
        
    elif method == 'pnp':
        from utils_inverse_problems.reconstruction_map_pnp import PnP_Recon_FBS, PnP_Recon_FISTA

        # load model
        model_type = opts.get('model_type', None)
        model_sigma = opts.get('model_sigma', None)
        if model_type is None or model_sigma is None:
            raise ValueError("Please provide a model_type and a model_sigma for the pnp model.")
        
        if model_type == "dncnn":
            sys.path += ['../../others/dncnn/model_dncnn/']
            from utils_dncnn import load_model as load_model_dncnn

            model = load_model_dncnn(model_sigma, device=device)
            model.eval()
            mode = "residual"

        elif model_type == "averaged_cnn":
            sys.path += ['../../others/dncnn/model_dncnn/', '../../others/averaged_cnn/model_averaged/']
            from utils_averaged_cnn import load_model as load_model_averaged_cnn

            mode = "direct"
            model = load_model_averaged_cnn(model_sigma, device=device)
            model.eval()

        else:
            raise ValueError(f"model_type {model_type} not supported")
        
        if modality == 'ct':
            if opts.get('n_hyperparameters', 1) == 2:
                def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                    with torch.no_grad():
                        x, psnr_, ssim_, n_iter = PnP_Recon_FBS(y, model, alpha=p2/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_init, **opts)
                    return(x, psnr_, ssim_, n_iter)
            else:
                def reconstruction_map(y, p1, x_gt=None, x_init=None):
                    with torch.no_grad():
                        x, psnr_, ssim_, n_iter = PnP_Recon_FBS(y, model, alpha=1.99/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_init, **opts)
                    return(x, psnr_, ssim_, n_iter)

        elif modality == 'mri':
            if opts.get('n_hyperparameters', 1) == 2:

                if model_type == "dncnn":

                    def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                        with torch.no_grad():
                            H, Ht = get_operators_mri(mask, smap, device=device)
                            op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                            x_zf = Ht(y)
                            x, psnr_, ssim_, n_iter = PnP_Recon_FISTA(y, model, alpha=p2/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_zf, **opts)
                        return(x, psnr_, ssim_, n_iter)
                    
                elif model_type == "averaged_cnn":

                    def reconstruction_map(y, mask, smap, p1, p2, x_gt=None, x_init=None):
                        with torch.no_grad():
                            H, Ht = get_operators_mri(mask, smap, device=device)
                            op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                            x_zf = Ht(y)
                            x, psnr_, ssim_, n_iter = PnP_Recon_FBS(y, model, alpha=p2/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_zf, **opts)
                        return(x, psnr_, ssim_, n_iter)

            else:

                if model_type == "dncnn":

                    def reconstruction_map(y, mask, smap, p1, x_gt=None, x_init=None):
                        with torch.no_grad():
                            H, Ht = get_operators_mri(mask, smap, device=device)
                            op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                            x_zf = Ht(y)
                            x, psnr_, ssim_, n_iter = PnP_Recon_FISTA(y, model, alpha=1.0/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_zf, **opts)
                        return(x, psnr_, ssim_, n_iter)
                    
                elif model_type == "averaged_cnn":

                    def reconstruction_map(y, mask, smap, p1, x_gt=None, x_init=None):
                        with torch.no_grad():
                            H, Ht = get_operators_mri(mask, smap, device=device)
                            op_norm = get_op_norm_mri(H, Ht, img_size=[mask.shape[2], mask.shape[3]], device=device)
                            x_zf = Ht(y)
                            x, psnr_, ssim_, n_iter = PnP_Recon_FBS(y, model, alpha=1.99/op_norm**2, lmbd=p1, H=H, Ht=Ht, x_gt=x_gt, mode=mode, x_init=x_zf, **opts)
                        return(x, psnr_, ssim_, n_iter)

        else:
            raise NotImplementedError

  
    elif method == 'acr':
        from utils_inverse_problems.reconstruction_map_acr import GD_Recon_torch
        sys.path += ['../../others/acr/model_acr/']
        from utils_acr import load_model as load_model_acr


        model = load_model_acr(device=device)
        model.eval()

        if modality == 'ct':
            def reconstruction_map(y, p1, p2, x_gt=None, x_init=None):
                x, psnr_, ssim_, n_iter = GD_Recon_torch(y, model, lmbd=p1, alpha=p2, H=H, Ht=Ht, x_gt=x_gt, x_init=x_init, **opts)
                return(x, psnr_, ssim_, n_iter)
            
        elif modality == 'mri':
            raise NotImplementedError
        
        else:
            raise NotImplementedError
       
            
    else:
        raise NotImplementedError
    
    return(reconstruction_map)
