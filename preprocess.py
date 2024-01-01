import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
import cv2

import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split



celebA_train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])
celebA_val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    ])

bdd100k_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])
bdd100k_val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    ])


# class CelebA(Dataset):
#     def __init__(self, data_path, label_path, mode, task_label = 'Gray_Hair'):   
        
#         self.images = np.load(data_path)['Image']
#         self.labels_list = np.load(label_path)
    
#         if mode == 'train':
#             self.transforms = celebA_train_transform 
#         else:
#             self.transforms = celebA_val_test_transform

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         image = Image.fromarray(image)
#         image = self.transforms(image)

#         target = self.labels_list[idx]

#         return idx, image, target

#     def __len__(self):
#         return len(self.images)

#     @property        
#     def labels(self):
#         return self.labels_list
class CelebA(Dataset):  ###read images
    def __init__(self, root, attr_csv,split_csv, mode, task_label = 'Gray_Hair'):   
        attributes = pd.read_csv(attr_csv)
        split = pd.read_csv(split_csv)

        mode_dict = {'train':0, 'val':1, 'test':2}
        selected_split = split[split['partition']==mode_dict[mode]]
        selected_attr = attributes[attributes.image_id.isin(selected_split.image_id)]
        
        images = selected_attr['image_id'].values
        self.images = [os.path.join(root, img) for img in images]
        self.targets = selected_attr[task_label].values
        self.targets[self.targets == -1] = 0
        
        print(f'{mode}, self.images:', len(self.images))
        print(f'{mode}, positive numbers:', self.targets.sum())
        if mode == 'train':
            self.transform = celebA_train_transform 
        else:
            self.transform = celebA_val_test_transform

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path)
        image = self.transform(image)

        target = self.targets[index]
        return index, image, target

    def __len__(self):
        return len(self.images)

    @property        
    def labels(self):
        return self.targets

class Bdd100k(Dataset):

    def __init__(self, data_path, attr_path, task='scene', pos_cls=0,
                        mode = 'train',val_ratio = 0.2, seed=0):
        #### scene 0 tunnel, weather 6 foggy
        self.images = np.load(data_path)['Image']

        self.attr = np.load(attr_path)
        self.task_map = {'weather':0, 'scene':1}
        self.labels_list = (self.attr[:,self.task_map[task]]==pos_cls).astype(int)
        self.mode = mode

        if self.mode!='test':
            X_train, X_val, y_train, y_val = train_test_split(self.images, self.labels_list, test_size=val_ratio, 
                                            random_state=42+seed,stratify=self.labels_list)
            if self.mode=='train':
                self.images= X_train
                self.labels_list = y_train
            elif self.mode=='val':
                self.images= X_val
                self.labels_list = y_val
            
        print(f'{mode}, self.images:', len(self.images))
        print(f'{mode}, positive numbers:', self.labels_list.sum())
        if self.mode == 'train':
            self.transforms = bdd100k_train_transform
        else:
            self.transforms = bdd100k_val_test_transform

    @property        
    def labels(self):
        return self.labels_list
    
    def __len__(self):
        return len(self.images) 
    
    def __getitem__(self,idx):

        image = self.images[idx]
        image = Image.fromarray(image)
        image = self.transforms(image)

        target = self.labels_list[idx]

        return idx, image, target



