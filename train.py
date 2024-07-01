import argparse
import json
import torch
from trainer import Trainer
import os
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

def main(args):

    config = json.load(open('config.json'))
    exp_dir = os.path.join(config['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed']) 
    torch.set_num_threads(8)
    
    trainer_inst = Trainer(json.load(open('config.json')), args.device)
    trainer_inst.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
    args = parser.parse_args()

    main(args)