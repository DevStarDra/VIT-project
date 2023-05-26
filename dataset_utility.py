import os
import glob
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ToTensor(object):
    def __call__(self, sample):
        sample = np.array(sample)
        return torch.tensor(sample, dtype=torch.float32)

class RAVENDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        img, target = self.X[index], self.y[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.X)


class dataset(Dataset):
    def __init__(self, root, dataset_type, fig_type='*', img_size=160, transform=None, train_mode=False):
        self.transform = transform
        self.img_size = img_size
        self.train_mode = train_mode

        self.file_names = []
        for filename in os.listdir(root):
            file_extension = os.path.splitext(filename)[1]
            #print(file_extension)
            if dataset_type in filename and file_extension == ".npz":
                self.file_names.append(root + "/" +  filename)
        
        if self.train_mode: 
            idx = list(range(len(self.file_names)))
            np.random.shuffle(idx)
            #self.file_names = [self.file_names[i] for i in idx[0:100000]]  # randomly select 100K samples for fast model training on large-scale dataset
        
    def __len__(self):
        return len(self.file_names)
    

    def __getitem__(self, idx):

        data = np.load(self.file_names[idx])

        image = data['image'].reshape(16, 160, 160)
        target = data['target']
        
        del data
        
        resize_image = image
        if self.img_size is not None:
            resize_image = []
            for idx in range(0, 3):
                resize_image.append(cv2.resize(image[idx, :], (self.img_size, self.img_size), interpolation = cv2.INTER_NEAREST))  

        if self.transform:
            resize_image = self.transform(resize_image)   
            target = torch.tensor(target, dtype=torch.long)
                    
        return resize_image, target
        
