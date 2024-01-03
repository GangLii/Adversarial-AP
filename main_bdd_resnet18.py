import torch
import numpy as np
import random

from preprocess import *
from utils import *
from config_bdd import conf
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
    train_data_path = root + "train_x.npz"
    train_attr_path = root + "train_y.npy"
    test_data_path = root + "test_x.npz"
    test_attr_path = root + "test_y.npy"

    model_name = 'resnet18'
    conf['loss_type'] = 'adv_x' ### please refer to 'Dict of Abbrevidations.txt'
    conf['loss_param'] = {'threshold': 0.6, 'gamma':(0.1,0.9), 'Lambda':0.8,} 
    out_path = './Released_results/{}/bdd_residential/results_{}'.format(model_name, conf['loss_type'])
    if not os.path.exists(out_path):   
        os.makedirs(out_path)

    logger = logger_init(out_path+'/log.log')
    for lr in [1e-3,1e-4,1e-5]:  
        if out_path is not None:
            fp = open(os.path.join(out_path, 'new_res.txt'), 'a')
            fp.write('new parameter \n')
            fp.close()
        for i in range(3):
            set_all_seeds(7777 + i)
            model = resnet18()
            train_dataset = Bdd100k(train_data_path, train_attr_path, task='scene', pos_cls=1, mode = 'train',seed=i)
            val_dataset = Bdd100k(train_data_path, train_attr_path, task='scene', pos_cls=1, mode = 'val',seed=i)
            test_dataset = Bdd100k(test_data_path, test_attr_path, task='scene', pos_cls=1, mode = 'test',seed=i)
            
            #def __init__(self, data_path, attr_path, task='weather', pos_cls=0,
            #                   mode = 'train',val_ratio = 0.2, seed=0):
                #### scene 0 :tunnel, 1:residential 
                #### weather 0:rainy, 5:partly cloudy
                
            ##***TO DO***#
            conf['lr'] = lr 
            
            logger.info(conf)
            logger.info(i)
            run_classification(i, train_dataset, val_dataset, test_dataset, model, 
                                conf, out_path, logger)
