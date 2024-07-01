import torch
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from dataloader.BSD500 import BSD500
#from dataloader.patch_dataset import PatchDataset
from R_network import RNet
from utils import metrics, utilities
import matplotlib.pyplot as plt

class Trainer:
    """
    """
    def __init__(self, config, device):

        self.config = config
        self.device = device
        self.sigma = config['sigma']

        # Prepare dataset classes
        train_dataset = BSD500(config['training_options']['train_data_file'])
        val_dataset = BSD500(config['training_options']['val_data_file'])

        print('Preparing the dataloaders')
        self.train_dataloader = DataLoader(train_dataset, batch_size=config["training_options"]["batch_size"], shuffle=True,\
                                             num_workers=config["training_options"]["num_workers"], drop_last=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        self.batch_size = config["training_options"]["batch_size"]

        print('Building the model')
        self.model = RNet(config['model_params'])
        self.model = self.model.to(device)
        #self.model.alpha_deq = self.model.alpha_deq.to(device)
        #self.model.theta = self.model.theta.to(device)
        self.model.eigenimage = self.model.eigenimage.to(device)
        #self.model.phi2grad.step_size = self.model.phi2grad.step_size.to(device)
        print(self.model)
        
        self.epochs = config["training_options"]['epochs']

        #
        #self.optimizer = torch.optim.Adam([{"params": self.model.W1.parameters(), "lr":1e-3}, {"params": self.model.W2.parameters(), "lr":1e-4},
              #                            {"params": self.model.W3.parameters(), "lr":5e-5}, {"params": self.model.psigrad_spline.parameters(), "lr":1e-5}, 
                  #                        {"params": self.model.Delta_psi, "lr":1e-5}, {"params": self.model.lmbda, "lr":1e-3}]
                      #                    , lr=config['training_options']['lr'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['training_options']['lr'])

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10], gamma=0.1)


        # CHECKPOINTS & TENSOBOARD
        run_name = config['exp_name']
        self.checkpoint_dir = os.path.join(config['log_dir'], config["exp_name"], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        config_save_path = os.path.join(config['log_dir'], config["exp_name"], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config['log_dir'], config["exp_name"], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

        self.total_training_step = 0
        #self.model.activation.step_size = self.model.activation.step_size.to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='sum').to(self.device) 
        

    def train(self):
        
        best_psnr = 0.0
        
        for epoch in range(self.epochs+1):
            
            #self.valid_epoch(epoch)
            self.train_epoch(epoch)
            psnr = self.valid_epoch(epoch)
            if psnr > best_psnr:
                best_psnr = psnr
                self.save_checkpoint(epoch)

        self.writer.flush()
        self.writer.close()

    def train_epoch(self, epoch):
        
        self.model.train()
        tbar = tqdm(self.train_dataloader, ncols=135, position=0, leave=True)
        log = {}
        for idx, data in enumerate(tbar):
            
            data = data.to(self.device)
            noisy_image = data + (self.sigma/255.0)*torch.randn(data.shape, device=self.device)

            self.optimizer.zero_grad()
            output = self.model(noisy_image)


            if self.config['training_options']['loss'] =='mse':
                loss = (self.criterion(output, data))/(self.batch_size)  
            elif self.config['training_options']['loss'] =='mae':
                loss = torch.nn.L1Loss()(output, data) 

            loss.backward()
            self.optimizer.step()
                
            log['train_loss'] = loss.real.detach().cpu().item()

            if (self.total_training_step) % max((len(tbar) // 1000), 1)  == 0:
                self.wrt_step = self.total_training_step * self.batch_size
                self.write_scalars_tb(log)
        

            tbar.set_description('T ({}) | TotalLoss {:.5f} |'.format(epoch, log['train_loss'])) 
            self.total_training_step += 1
        self.scheduler.step()



    def valid_epoch(self, epoch):
        #self.model.niter = 50
        self.model.eval()
        loss_val = 0.0
        psnr_val = []
        ssim_val = []

        tbar_val = tqdm(self.val_dataloader, ncols=130, position=0, leave=True)
        
        with torch.no_grad():
            for data in tbar_val:
                data = data.to(self.device)
                noisy_image = data + (self.sigma/255.0)*torch.randn(data.shape, device=self.device)

                output = self.model(noisy_image)
                loss = self.criterion(output, data)

                loss_val = loss_val + loss.cpu().item()
                out_val = torch.clamp(output, 0., 1.)

                psnr_val.append(utilities.batch_PSNR(out_val, data, 1.))
                ssim_val.append(utilities.batch_SSIM(out_val, data, 1.))
            
            # PRINT INFO
            loss_val = loss_val/len(self.val_dataloader)

            # METRICS TO TENSORBOARD
            self.wrt_mode = 'val'
            self.writer.add_scalar(f'{self.wrt_mode}/Test PSNR Mean', np.mean(psnr_val), epoch)
            self.writer.add_scalar(f'{self.wrt_mode}/Test SSIM Mean', np.mean(ssim_val), epoch)
        self.model.train()
        
        return np.mean(psnr_val)


    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint_best_epoch.pth'
        torch.save(state, filename)