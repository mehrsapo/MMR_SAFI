import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random


class PatchDataset(Dataset):

    def __init__(self, data_file, data_augmentation=False):
        super(Dataset, self).__init__()
        self.data_augmentation = data_augmentation
        self.data_file = data_file
        self.dataset = None
        with h5py.File(self.data_file, 'r') as file:
            self.keys_list = list(file.keys())
            random.shuffle(self.keys_list)
        self.nb_img = len(self.keys_list)


    def __len__(self):
        if self.data_augmentation:
            return self.nb_img*8
        else:
            return self.nb_img


    def __getitem__(self, idx):
        if self.data_augmentation: img_idx = idx % self.nb_img
        else: img_idx = idx

        if self.dataset is None:
            self.dataset = h5py.File(self.data_file, 'r')
        data = torch.Tensor(np.array(self.dataset[self.keys_list[img_idx]]))
        data = data[:,:8*(data.shape[1]//8), :8*(data.shape[2]//8)]
        if idx // self.nb_img == 1:
            data = torch.flip(data, [1])
        elif idx // self.nb_img == 2:
            data = torch.flip(data, [2])
        elif idx // self.nb_img == 1:
            data = torch.rot90(data, 1, [1, 2])
        elif idx // self.nb_img == 2:
            data = torch.rot90(data, 2, [1, 2])
        elif idx // self.nb_img == 3:
            data = torch.rot90(data, 3, [1, 2])
        elif idx // self.nb_img == 6:
            data = torch.flip(torch.rot90(data, 1, [1, 2]), [1])
        elif idx // self.nb_img == 7:
            data = torch.flip(torch.rot90(data, 1, [1, 2]), [2])
        return data