import torch
import numpy as np
import random

from .utils.preprocess import *
from .utils.utils import *
from .utils.imbalanced_cifar import *
from .configs.config_cifar import conf
from train_eval import *
import argparse

parser = argparse.ArgumentParser(description='Adversarial AP Training')
parser.add_argument('--method', default='AdAP_LPN', type=str)
parser.add_argument('--th', default=0.6, type=float, help='threshold for squared hinge surrogate loss')
parser.add_argument('--gamma1', default=0.1, type=float)
parser.add_argument('--gamma2', default=0.9, type=float)
parser.add_argument('--Lambda', default=0.8, type=float)


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
if __name__ == '__main__':
    set_all_seeds(7777)
    args = parser.parse_args()

    model_name = 'resnet18'
    conf['loss_type'] = args.method
    conf['loss_param'] = {'threshold': args.th, 'gamma':(args.gamma1,args.gamma2),
                          'Lambda':args.Lambda} 

    for cls in range(10):
        out_path = './Released_results/{}/cifar100_coarsecls{}/results_{}'.format(model_name, cls, conf['loss_type'])
        if not os.path.exists(out_path):   
            os.makedirs(out_path)
        
        # CUDA_VISIBLE_DEVICES=7  nohup python ../main_cifar100_resnet18.py &
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
                train_dataset = OvA_CIFAR100(root='./data', download=True, transform = transform_train, mode = 'train', pos_class=pos_class, seed=i )
                val_dataset = OvA_CIFAR100(root='./data', download=True, transform=transform_val, mode = 'valid', pos_class=pos_class, seed=i)
                test_dataset = OvA_CIFAR100(root='./data', train=False, download=True, transform=transform_val, mode = 'test', pos_class=pos_class, seed=i)
                

                ##***TO DO***#
                conf['lr'] = lr 

                logger.info(pos_class)
                logger.info(conf)
                logger.info(i)

                run_classification(i, train_dataset, val_dataset, test_dataset, model, 
                                    conf, out_path, logger)