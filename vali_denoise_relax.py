from validate_coarse_to_fine import * 
from dataloader.BSD500 import BSD500
import torch
from R_network_relax import RNet
import json
from matplotlib import pyplot as plt
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim



valid_dataset = BSD500('data/val.h5')

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

sigma = 5.0   

exp = "/home/pourya/mm_final/exps/sigma5/64_2_7_1e-3_relaxed"
path_ckp = exp + "/checkpoints/checkpoint_best_epoch.pth"
path_config = exp + "/config.json"
device = 'cuda:3'
config = json.load(open(path_config))
ckp = torch.load(path_ckp, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

model = RNet(config['model_params'])
model.to(device)
model.load_state_dict(ckp['state_dict'])
model.eval()


model.W1.W1.weight.data = model.zeromean(model.W1.W1.weight)
for i, _ in enumerate(model.W1.W1s): 
    model.W1.W1s[i].weight.data = model.zeromean(model.W1.W1s[i].weight)


model.eigenimage = model.eigenimage.to(device)
print(model.lmbda)
lip, model.eigenimage = model.W1.cal_lip(model.eigenimage, 50)
model.alpha = 1 / lip


def safi_denoise(y):
    s = 1e-3
    n = 5
    e = 1e-5
    q = (e/s) ** (1/n)
    tols = [s * q**(i) for i in range(n+1)]
    n_out = 10
    for _ in range(n_out-(n+1)):
        tols.append(e)
    with torch.no_grad(): 
        n_in = 100  
        c_k = model.prox_denoise_no_mask(y, y, n_in, tols[0], check_tol=True)
        for it in range(n_out - 1):   
            model.cal_mask(c_k)
            c_k = model.prox_denoise_with_mask(y, c_k, n_in, tols[it+1], check_tol=True)

    return c_k



def score(lmbda):
    model.lmbda = torch.nn.Parameter(torch.ones(1, 1).to(device) * lmbda)
    psnrs = list(); ssims = list()
    i = 0
    for img in valid_dataset:
        i = i + 1
        if True:
            gt = img.to(device)[None, :, :, :]
            noisy_image = (img.to(device) + (sigma/255.0)*torch.randn(img.shape, device=device))[None, :, :, :]
            denoised = safi_denoise(noisy_image)
            psnr_ = psnr(denoised, gt, data_range=1).item()
            ssim_ = ssim(denoised, gt, data_range=1).item()
            psnrs.append(psnr_)
            ssims.append(ssim_)

            print(i, psnr_, ssim_)
    
    return (np.mean(np.array(psnrs)), np.mean(np.array(ssim_)), 200)




val = ValidateCoarseToFine(score, 'denoise_val_results', 'sigma_5_relaxed_valid', p1_init=model.lmbda.item(), freeze_p2=True)


with torch.no_grad():
    val.run()
