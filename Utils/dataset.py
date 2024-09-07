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

Note: Class must be a name, Eg: Cat, Dot etc.
'''

import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    '''
    Custom Dataset for CLIP Model.
    DatasetObject[Index] Output
        -> (Image, Text, Class)
    '''
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.data = []

        self.data.extend(
            (os.path.join(self.parent_dir, class_dir, file), f'An image of a {class_dir}.')
            for class_dir in os.listdir(self.parent_dir)
            for file in os.listdir(os.path.join(self.parent_dir, class_dir))
        )

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        record = self.data[index]

        img = record[0]
        text = record[1]

        return img, text