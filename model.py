# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:04:11 2020

@author: kasy
"""

import logging as log
import os
from argparse import ArgumentParser
from collections import OrderedDict

import pandas as pd
from sklearn.utils import shuffle
from PIL.Image import BICUBIC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import *
import math

from template import LightningTemplateModel
from data import ImageDataset
from utils_model import *
from pytorch_lightning.core import LightningModule


class ResNet(LightningModule):
    
    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(ResNet, self).__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        #self.example_input_array = torch.rand(10, 3, 224, 224)

        # build model
        self.__build_model(baseWidth=4, cardinality=32, layers=[3, 4, 23, 3], num_classes=6)
    
    def __dataloader(self, train_mode):
        self.prepare_data()
        batch_size = self.hparams.batch_size
        num_workers = 4
        if train_mode == 'train':
            transform = self.train_transform
            dataset = ImageDataset(root='./data/train/',
                             path_list=self.train_x, targets=self.train_y, transform=transform)
            
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                          shuffle=True, drop_last=True, pin_memory=True)
        
        elif train_mode == 'valid':
            dataset = ImageDataset(root='./data/train/',
                                  path_list=self.valid_x, targets=self.valid_y, transform=self.test_transform)
            
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                          shuffle=False, drop_last=False, pin_memory=True)
        else:
            dataset = ImageDataset(root='./data/test/',
                            path_list=self.test_x, transform=self.test_transform)
            
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                          shuffle=False, drop_last=False, pin_memory=True)
                 
        return loader
  
    def prepare_data(self):
        resolution = 224
        img_stats  = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        
        self.classes = ('No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR')
        
        df_train = pd.read_csv('./data/Train_label.csv')
        df_test = pd.read_csv('./data/Test_label.csv')
        
        x = df_train['id_code']
        y = df_train['diagnosis']
        
        x, y = shuffle(x, y)
        n_classes = int(y.max()+1)
        class_weights = len(y) / df_train.groupby('diagnosis').size().values.ravel()  # we can use this to balance our loss function
        class_weights *= n_classes / class_weights.sum()
        self.class_weights = class_weights
        
        train_x, valid_x, train_y, valid_y = train_test_split(x.values, y.values, test_size=0.20, stratify=y, random_state=42)
        self.test_x = df_test.id_code.values
        
        self.train_x = train_x
        self.valid_x = valid_x
        self.train_y = train_y
        self.valid_y = valid_y
        
        self.train_transform = Compose([
            Resize([resolution]*2, BICUBIC),
            ColorJitter(brightness=0.05, contrast=0.05, saturation=0.01, hue=0),
            RandomAffine(degrees=15, translate=(0.01, 0.01), scale=(1.0, 1.25), fillcolor=(0,0,0), resample=BICUBIC),
            RandomHorizontalFlip(),
        #     RandomVerticalFlip(),
            ToTensor(),
            Normalize(*img_stats)
        ])
        
        self.test_transform = Compose([
            Resize([resolution]*2, BICUBIC),
            ToTensor(),
            Normalize(*img_stats)
        ])
    
    def train_dataloader(self, train_mode='train'):
        log.info('Training data loader called.')
        return self.__dataloader(train_mode=train_mode)

    def val_dataloader(self, train_mode='valid'):
        log.info('Validation data loader called.')
        return self.__dataloader(train_mode=train_mode)

    def test_dataloader(self, train_mode='test'):
        log.info('Test data loader called.')
        return self.__dataloader(train_mode=train_mode)
    
    def __build_model(self, baseWidth, cardinality, layers, num_classes):
        """
        Layout model
        :return:
        """
        block = Bottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
    
        
        self.avgpool = nn.AvgPool2d(7)      
        
        #512 * block.expansion original
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def loss(self, labels, logits):
        targets = labels
        inputs = logits
        
        def cont_kappa(input, targets, activation=None):
            ''' continuos version of quadratic weighted kappa '''
            n = len(targets)
            y = targets.float().unsqueeze(0)
            pred = input.float().squeeze(-1).unsqueeze(0)
            if activation is not None:
                pred = activation(pred)
            wo = (pred - y)**2
            we = (pred - y.t())**2
            return 1 - (n * wo.sum() / we.sum())

        kappa_loss = lambda pred, y: 1 - cont_kappa(pred, y)  # from 0 to 2 instead of 1 to -1
        activation = lambda y: y  # no-op
        
        def Focal_loss(inputs, targets, alpha=None, gamma=2., reduction='mean'):
            
            alpha = torch.tensor(alpha)
            
            alpha = alpha.type(inputs.type(), non_blocking=True) # fix type and device
    
            CE_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-CE_loss)
            F_loss = alpha[targets] * (1-pt)**gamma * CE_loss
    
            if reduction == 'sum':
                return F_loss.sum()
            elif reduction == 'mean':
                return F_loss.mean()
            return F_loss
        
        second_mult=0.5
        
        loss  = Focal_loss(inputs[...,:-1], targets, alpha=self.class_weights)  # focal loss
        second_loss = kappa_loss
        loss += second_mult * second_loss(inputs[...,-1], targets.float())
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch
        #x = x.view(x.size(0), -1)

        y_hat = self.forward(x)

        # calculate loss, labels, logits
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output
    
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch
        #x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat[..., :-1], dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        print('Val acc {0:.4f}'.format(val_acc_mean))
        return result
    
    
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--hidden_dim', default=50000, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=32, type=int)
        return parser
        