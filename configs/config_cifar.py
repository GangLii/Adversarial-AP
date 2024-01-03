
"""
Configuration file for cifar
"""

conf = {}

######################################################################################################################
# Settings for training
##    'epochs': maximum training epochs
##    'lr': starting learning rate
##    'lr_decay_factor': learning rate decay factor
##    'lr_decay_step': learning rate decay at each step
##    'weight_decay': l2 regularizer term
##    'batch_size': training batch_size
######################################################################################################################


# cifar10
conf['batch_size'] = 128
conf['epochs'] = 60

conf['lr'] = 0.001
conf['lr_decay_factor'] = 0.1
conf['lr_decay_step'] = [int(conf['epochs']*0.5),int(conf['epochs']*0.75)]

conf['weight_decay'] = 2e-4  

conf['surr_loss'] = 'sqh'
conf['loss_param'] = None
conf['pre_train'] = None

conf['posNum'] = conf['batch_size']//2