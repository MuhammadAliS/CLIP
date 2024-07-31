'''
Module load the dataset from parent directory.
Directory format,

    |->ParentDirectory
        |->Class-1
            |->Image-1
            |->Image-2
            ...
        |->Class-2
            |->Image-1
            |->Image-2
            ...
        ...

Note: Class must be in int format. Eg: 0, 1, 2 etc.
'''

import os
import random

import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

class CustomDataset(Dataset):
    '''
    Custom Dataset for CLIP Model.
    DatasetObject[Index] Output
        -> (Image, Text, Class)
    '''
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.data = []
        self.transforms = v2.Compose(
            [v2.RandomResizedCrop(size=(28,28), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def __len__(self):

        counter = 0
        for class_dir in os.listdir(self.parent_dir):
            f_path = os.path.join(self.parent_dir, class_dir)
            text = f'Image of number {int(class_dir)}'
            for _ in os.listdir(f_path):
                self.data.append((os.path.join(f_path,_),
                                  text,
                                  int(class_dir)))
                counter+=1

        random.shuffle(self.data)   
        return counter
    
    def __getitem__(self, index):
        record = self.data[index]

        img = self.transforms(read_image(record[0], 
                                         mode = ImageReadMode.UNCHANGED))

        text = record[1]
        label = record[2]

        return img, text, label