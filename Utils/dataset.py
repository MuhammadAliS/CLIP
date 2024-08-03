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
import clip
from PIL import Image
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class CustomDataset(Dataset):
    '''
    Custom Dataset for CLIP Model.
    DatasetObject[Index] Output
        -> (Image, Text, Class)
    '''
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.data = []
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
        self.dataset_length = counter

    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        record = self.data[index]

        img = preprocess(Image.open(record[0]))
        text = record[1]
        label = record[2]

        return img, text, label