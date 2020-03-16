# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:20:58 2020

@author: kasy
"""



from torch.utils.data import Subset
import torchvision.utils as vutils
import os
import torch

import pandas as pd
from sklearn.utils import shuffle
from PIL import Image
from torchvision.transforms import *

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, root, path_list, targets=None, transform=None, extension='.jpg'):
        super().__init__()
        self.root = root
        self.path_list = path_list
        self.targets = targets
        self.transform = transform
        self.extension = extension
        if targets is not None:
            assert len(self.path_list) == len(self.targets)
#            self.g_targets = torch.LongTensor(targets[0])
#            self.e_targets = torch.LongTensor(targets[1])
            self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.path_list[index]
        sample = Image.open(os.path.join(self.root, path+self.extension))
        if self.transform is not None:
            sample = self.transform(sample)

        #self.ones = np.zeros

        if self.targets is not None:
            #return sample, self.g_targets[index], self.e_targets[index]
            return sample, self.targets[index]
        else:
            #return sample, torch.LongTensor([]), torch.LongTensor([])
            return sample, torch.LongTensor([])

    def __len__(self):
        return len(self.path_list)