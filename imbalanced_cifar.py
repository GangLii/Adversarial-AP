import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


class OvA_CIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    coarse = False

    def __init__(self, root, train=True,transform=None, target_transform=None, download=False,
                pos_class=1, mode = 'train', train_ratio = 0.8, seed=0):
        super(OvA_CIFAR10, self).__init__(root, train, transform, target_transform, download)
        
        print('pos_class',pos_class)
        img_num_per_cls = len(self.data) / self.cls_num
        new_data, new_targets = [], []

        if self.coarse == True:
            self.targets = self.coarse_labels[self.targets]
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        for the_class in classes:
            idx = np.where(targets_np == the_class)[0]
            np.random.seed(seed)
            np.random.shuffle(idx)
            ### data part
            if mode == 'train':
                selec_idx = idx[:int(img_num_per_cls*train_ratio)]
            elif mode == 'valid':
                selec_idx = idx[int(img_num_per_cls*train_ratio):]
            elif mode == 'test':
                selec_idx = idx
            else: print('data split error')
            
            ###postive or negative
            if the_class == pos_class:
                # if self.cls_num == 10:
                #     pos_num = int(len(selec_idx)*0.1)
                #     selec_idx = selec_idx[:pos_num]
                new_targets.extend([1] * len(selec_idx))
            else:
                new_targets.extend([0] * len(selec_idx))
            
            new_data.append(self.data[selec_idx, ...])

        self.data = np.vstack(new_data)  
        self.targets = new_targets
        print(f'{mode}, self.data:', len(self.data))
        print(f'{mode}, positive numbers:', sum(self.targets))
    
    @property        
    def labels(self):
        return self.targets

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return index, img, target

class OvA_CIFAR100(OvA_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 20
    coarse = True

    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

    def __init__(self, root, train=True, transform=None, target_transform=None,
             download=False, mode = 'train', pos_class=1, seed=0 ):
        OvA_CIFAR10.__init__(self, root, train=train,transform=transform, target_transform=target_transform,
                              download=download, pos_class=pos_class, mode = mode, train_ratio = 0.8, seed=seed )



