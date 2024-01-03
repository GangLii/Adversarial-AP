import torch
import numpy as np
import random

from .utils.preprocess import *
from .utils.utils import *
from .utils.imbalanced_cifar import *
from .configs.config_celebA import conf
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

root = 'path to root' ### PATH TO THE DATA FOLDER
image_path = root+'img_align_celeba/img_align_celeba'
attr_path = root+'list_attr_celeba.csv'
parti_path = root+'list_eval_partition.csv'   
    
    
    
if __name__ == '__main__':
    set_all_seeds(7777)
    args = parser.parse_args()

    model_name = 'resnet18'
    conf['loss_type'] = args.method
    conf['loss_param'] = {'threshold': args.th, 'gamma':(args.gamma1,args.gamma2),
                          'Lambda':args.Lambda} 
    
    out_path = './Released_results/{}/CelebA_mustache/results_{}'.format(model_name, conf['loss_type'])
    if not os.path.exists(out_path):   
        os.makedirs(out_path)

    logger = logger_init(out_path+'/log.log')
    for lr in [1e-3,1e-4,1e-5]:         ### TODO
        if out_path is not None:
            fp = open(os.path.join(out_path, 'new_res.txt'), 'a')
            fp.write('new parameter \n')
            fp.close()

        for i in range(3):
            set_all_seeds(7777+i)
            model = resnet18()
            train_dataset = CelebA(image_path, attr_path,parti_path, mode = 'train', task_label = 'Mustache' )
            val_dataset = CelebA(image_path, attr_path,parti_path, mode = 'val', task_label = 'Mustache')
            test_dataset = CelebA(image_path, attr_path,parti_path,  mode = 'test', task_label = 'Mustache')
            ### Gray_Hair    Mustache 

            ##***TO DO***#
            conf['lr'] = lr 

            logger.info(conf)
            logger.info(i)
            run_classification(i, train_dataset, val_dataset, test_dataset, model, 
                                conf, out_path, logger)
