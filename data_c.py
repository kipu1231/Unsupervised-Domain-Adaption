import os
from os import listdir
import csv
import numpy as np
import torch
import random
import pandas as pd

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Mnist(Dataset):
    def __init__(self, args, mode, visualization=False):
        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir

        if self.mode == 'val' or self.mode == "train":
            self.img_dir = os.path.join(self.data_dir, 'mnistm/' + self.mode)
            if visualization:
                self.img_dir = os.path.join(self.data_dir, 'mnistm/test')
        elif self.mode == "test":
            self.img_dir = self.data_dir

        ''' read the data list '''
        if self.mode == 'val' or self.mode == "train":
            csv_path = os.path.join(self.data_dir,  'mnistm/' + self.mode + '.csv')

            if visualization:
                csv_path = os.path.join(self.data_dir, 'mnistm/test.csv')

            with open(csv_path, newline='') as csvfile:
                next(csvfile)
                self.data = list(csv.reader(csvfile))

            ''' set up image path '''
            for d in self.data:
                d[0] = os.path.join(self.img_dir, d[0])
                d[1] = int(d[1])

        elif self.mode == 'test':
            helper = []
            for f in sorted(listdir(self.img_dir)):
                img_path = os.path.join(self.data_dir, f)
                helper.append(img_path)

            array = np.array(helper)
            self.data = array


        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                transforms.Normalize(MEAN, STD)
            ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                transforms.Normalize(MEAN, STD)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' get data '''
        if self.mode == 'val' or self.mode == "train":
            img_path, cls = self.data[idx]
        elif self.mode == 'test':
            img_path = self.data[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        ''' distinguish wether data is target or source '''
        if self.mode == 'val' or self.mode == "train":
            return self.transform(img), cls
        elif self.mode == 'test':
            return self.transform(img), img_path


class Svhn(Dataset):
    def __init__(self, args, mode, visualization=False):
        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir

        if self.mode == 'val' or self.mode == "train":
            self.img_dir = os.path.join(self.data_dir, 'svhn/' + self.mode)
            if visualization:
                self.img_dir = os.path.join(self.data_dir, 'svhn/test')
        elif self.mode == "test":
            self.img_dir = self.data_dir

        ''' read the data list '''
        if self.mode == 'val' or self.mode == "train":
            csv_path = os.path.join(self.data_dir, 'svhn/' + self.mode + '.csv')
            if visualization:
                csv_path = os.path.join(self.data_dir, 'svhn/test.csv')

            with open(csv_path, newline='') as csvfile:
                next(csvfile)
                self.data = list(csv.reader(csvfile))

            ''' set up image path '''
            for d in self.data:
                d[0] = os.path.join(self.img_dir, d[0])
                d[1] = int(d[1])

        elif self.mode == 'test':
            helper = []
            for f in sorted(listdir(self.img_dir)):
                img_path = os.path.join(self.data_dir, f)
                helper.append(img_path)

            array = np.array(helper)
            self.data = array

        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((28,28)),
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                transforms.Normalize(MEAN, STD)
            ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                transforms.Normalize(MEAN, STD)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' get data '''
        if self.mode == 'val' or self.mode == "train":
            img_path, cls = self.data[idx]
        elif self.mode == 'test':
            img_path = self.data[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        ''' distinguish wether data is target or source '''
        if self.mode == 'val' or self.mode == "train":
            return self.transform(img), cls
        elif self.mode == 'test':
            return self.transform(img), img_path


class DataC(Dataset):
    def __init__(self, args, mode):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, self.mode + '/img/')
        self.label_dir = os.path.join(self.data_dir, self.mode + '/seg/')

        ''' read the data list '''
        list = []
        for f in sorted(listdir(self.img_dir)):
            list.append(f)
            list.append(f)

        array = np.array(list)
        array = array.reshape(-1, 2)
        self.data = array

        help_list = []

        ''' set up image path '''
        for d in self.data:
            im_path = os.path.join(self.img_dir, d[0])
            lb_path = os.path.join(self.label_dir, d[1])
            help_list.append(im_path)
            help_list.append(lb_path)

        help_array = np.array(help_list)
        help_array = help_array.reshape(-1, 2)
        self.data = help_array

        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                #transforms.Normalize(MEAN, STD)
            ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                #transforms.Normalize(MEAN, STD)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path, lbl_path = self.data[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        cls = Image.open(lbl_path).convert('L')

        if random.random() > 0.6:
            angle = random.randint(-30, 30)
            img.rotate(angle)
            cls.rotate(angle)

        cls = np.array(cls)
        cls = torch.from_numpy(cls)
        cls = cls.long()

        return self.transform(img), cls
