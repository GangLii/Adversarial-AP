
"""
Configuration file for Bdd100k
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

conf['batch_size'] = 128
conf['epochs'] = 32

conf['lr'] = 0.001
conf['lr_decay_factor'] = 0.1
conf['lr_decay_step'] = [int(conf['epochs']*0.5),int(conf['epochs']*0.75)]

conf['weight_decay'] = 1e-5  

conf['surr_loss'] = 'sqh'
conf['loss_param'] = None
conf['pre_train'] = None

conf['posNum'] = conf['batch_size']//2
