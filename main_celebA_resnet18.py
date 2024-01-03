import torch
import numpy as np
import random

from preprocess import *
from utils import *
from config_celebA import conf
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
    
    
if __name__ == '__main__':
    set_all_seeds(7777)
    root = 'path to root'
    image_path = root+'img_align_celeba/img_align_celeba'
    attr_path = root+'list_attr_celeba.csv'
    parti_path = root+'list_eval_partition.csv'

    model_name = 'resnet18'
    conf['loss_type'] = 'adv_x' ### please refer to 'Dict of Abbrevidations.txt'
    conf['loss_param'] = {'threshold': 0.6, 'gamma':(0.1, 0.9), 'Lambda':0.8}  
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
