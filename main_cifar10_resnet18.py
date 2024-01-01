import torch
import numpy as np
import random

from preprocess import *
from utils import *
from config_cifar import conf
from train_eval import *
from imbalanced_cifar import *



def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_all_seeds(7777)

model_name = 'resnet18scratch'
conf['loss_type'] = 'adv_t' ### please refer to 'Dict of Abbrevidations.txt'
conf['loss_param'] = {'threshold': 0.6, 'gamma':(0.1,0.1), 'Lambda':0.01} 

## 0 2 4
for cls in range(10):
    out_path = './Released_results/{}/cifar10_cls{}/results_{}g01lm001'.format(model_name, cls, conf['loss_type'])
    if not os.path.exists(out_path):   
        os.makedirs(out_path)
        
    # CUDA_VISIBLE_DEVICES=7  nohup python ../main_cifar10_resnet18.py &
    logger = logger_init(out_path+'/log.log')
    for lr in [1e-3,1e-4,1e-5]:  
        if out_path is not None:
            fp = open(os.path.join(out_path, 'new_res.txt'), 'a')
            fp.write('new parameter \n')
            fp.close()
        for i in range(3):
            set_all_seeds(7777 + i)
            model = resnet18()  
            pos_class = cls
            train_dataset = OvA_CIFAR10(root='../data', download=True, transform = transform_train, mode = 'train', pos_class=pos_class, seed=i )
            val_dataset = OvA_CIFAR10(root='../data', download=True, transform=transform_val, mode = 'valid', pos_class=pos_class, seed=i)
            test_dataset = OvA_CIFAR10(root='../data', train=False, download=True, transform=transform_val, mode = 'test', pos_class=pos_class, seed=i)
            

            ##***TO DO***#
            conf['lr'] = lr 
            conf['pre_train'] = None #f'./Released_results/resnet18scratch/cifar10_cls{cls}/results_ce/{i}_best_0_lr0.001.ckpt' # +str(i)

            logger.info(pos_class)
            logger.info(conf)
            logger.info(i)

            run_classification(i, train_dataset, val_dataset, test_dataset, model, 
                                conf, out_path, logger)
