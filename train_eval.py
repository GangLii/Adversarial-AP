import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from new_loss import DataSampler, AUPRC_Loss, TRADES, MART, ADV_AP_V2, ADV_TRADES, MinM_AP, ADV_AP_X
from attacks import ranking_attack_v1, PGD_attack, TRADES_attack, AP_attack


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### This is run function for classification tasks
def run_classification(i, train_dataset, val_dataset, test_dataset, model, conf,
         save_dir,logger, imb_factor = 0.02):


    model = model.to(device)
    if conf['pre_train'] is not None:
        logger.info('we are loading pretrain model')
        state_key = torch.load(conf['pre_train'])
        logger.info('pretrain model is loaded from {} epoch'.format(state_key['epoch']))
        filtered = {k:v for k,v in state_key['model'].items() if 'fc' not in k}
        model.load_state_dict(filtered, False)
        model.fc.reset_parameters()


    if conf['loss_type']  in ['ce','pgd']:
        criterion = torch.nn.BCELoss()
    elif conf['loss_type'] in ['adv_2','adv_2m']:
        criterion = ADV_AP_V2(margin=conf['loss_param']['threshold'],
                                gamma = conf['loss_param']['gamma'], 
                                Lambda = conf['loss_param']['Lambda'],
                                data_length = len(train_dataset))
    elif conf['loss_type'] in ['adv_x']:
        criterion = ADV_AP_X(margin=conf['loss_param']['threshold'],
                                gamma = conf['loss_param']['gamma'], 
                                Lambda = conf['loss_param']['Lambda'],
                                data_length = len(train_dataset))
    elif conf['loss_type'] in ['adv_t']:
        criterion = ADV_TRADES(margin=conf['loss_param']['threshold'],
                                gamma = conf['loss_param']['gamma'][0], 
                                Lambda = conf['loss_param']['Lambda'],
                                data_length = len(train_dataset))
    elif conf['loss_type'] in ['ap']:
        criterion = AUPRC_Loss(margin=conf['loss_param']['threshold'],
                                gamma = conf['loss_param']['gamma'][0], 
                                data_length = len(train_dataset))
    elif conf['loss_type'] in ['mm_ap']:
        criterion = MinM_AP(margin=conf['loss_param']['threshold'],
                                gamma = conf['loss_param']['gamma'][0], 
                                data_length = len(train_dataset))
    elif conf['loss_type'] in ['tra']:
        criterion = TRADES(Lambda = conf['loss_param']['Lambda'])
    elif conf['loss_type'] in ['mart']:
        criterion = MART(Lambda = conf['loss_param']['Lambda'])

   
    optimizer = Adam(model.parameters(), lr=conf['lr'], weight_decay=conf['weight_decay'])

    valTrain_loader = DataLoader(train_dataset, conf['batch_size'], shuffle=False, num_workers=6, pin_memory=False)
    val_loader = DataLoader(val_dataset, conf['batch_size'], shuffle=False, num_workers=6, pin_memory=False)
    test_loader = DataLoader(test_dataset, conf['batch_size'], shuffle=False, num_workers=6, pin_memory=False)

    if conf['loss_type'] in ['ce', 'tra', 'pgd','mart']:
        train_loader = DataLoader(train_dataset, conf['batch_size'], shuffle=True, drop_last=False, num_workers=6,
                                    pin_memory=False)
    else:

        train_loader = DataLoader(train_dataset, batch_size=conf['batch_size'], 
        sampler=DataSampler(train_dataset.labels, conf['batch_size'], pos_num=conf['posNum']), 
        num_workers=6, pin_memory=False)

    ###scores
    ###[clean_ap, adv_ap]
    best_val_scores = [0,0]
    final_test_scores = [0,0]
    best_test_scores = [0,0]


    train_APs_clean = []
    val_APs_adv= []

    for epoch in range(1, conf['epochs'] + 1):

        avg_train_loss = train_classification(model, optimizer, train_loader, criterion, conf['loss_type'],epoch)

        valTrain_ap_result = valTrain_classification(model, valTrain_loader)
        val_ap_results = val_classification(model, val_loader)
        # val_ap_results = [val_ap_clean,val_ap_adv]
        if epoch == conf['epochs']:
            test_ap_results = val_classification(model, test_loader)
            # test_ap_results =[test_ap_clean,test_ap_adv]
        else:
            test_ap_results = [0,0]

        for k in range(len(best_val_scores)):

            if best_val_scores[k] <= val_ap_results[k]:
                best_val_scores[k] = val_ap_results[k]
                if (k==1 and conf['loss_type'] not in ['ce','ap']) or (k==0
                        and conf['loss_type'] in ['ce','ap']) :  ### we focus on robust ap
                    if epoch != conf['epochs']:
                        test_ap_results = val_classification(model, test_loader)
                    final_test_scores = test_ap_results
                    if save_dir is not None:
                        lr = conf['lr']
                        torch.save({'model':model.state_dict(), 'epoch':epoch}, os.path.join(save_dir, f'{i}_best_{k}_lr{lr}.ckpt'))
                elif conf['loss_type']=='ce' and save_dir is not None:
                    lr = conf['lr']
                    torch.save({'model':model.state_dict(), 'epoch':epoch}, os.path.join(save_dir, f'{i}_best_{k}_lr{lr}.ckpt'))

            if best_test_scores[k] <= test_ap_results[k]:
                best_test_scores[k] = test_ap_results[k]


        train_APs_clean.append(valTrain_ap_result)
        val_APs_adv.append(val_ap_results[1])
        logger.info('Epoch: {:03d}, Training Loss: {:.6f}, Train clean_AP: {:.4f}, Val clean_AP : {:.4f}, Best Val clean_AP: {:.4f}, \n \
            Test clean_AP: {:.4f},      Final clean_AP: {:.4f},       Best Test clean_AP: {:.4f}, \n \
            Val adv_AP : {:.4f}, Best Val adv_AP: {:.4f}, \n \
            Test adv_AP: {:.4f},      Final adv_AP: {:.4f},       Best Test adv_AP: {:.4f}, '
              .format(epoch, avg_train_loss, valTrain_ap_result, val_ap_results[0], best_val_scores[0], 
              test_ap_results[0], final_test_scores[0], best_test_scores[0],
              val_ap_results[1], best_val_scores[1],
              test_ap_results[1], final_test_scores[1], best_test_scores[1]))


        if epoch in conf['lr_decay_step']:
            for param_group in optimizer.param_groups:
                param_group['lr'] = conf['lr_decay_factor'] * param_group['lr']
                logger.info('learning rate decays to{}'.format(param_group['lr']))
        # if epoch == 10 and conf['loss_type'] == 'adv_2':
        #     logger.info('adjust lambda from {} to {}'.format(criterion.Lambda, conf['loss_param']['Lambda']))
        #     criterion.Lambda = conf['loss_param']['Lambda']
            
            
    if save_dir is not None:

        fp = open(os.path.join(save_dir, 'new_res.txt'), 'a')
        for score in best_val_scores:
            fp.write('{:.4f},'.format(score))
        for score in final_test_scores:
            fp.write('{:.4f},'.format(score))
        for score in test_ap_results:
            fp.write('{:.4f},'.format(score))
        fp.write('\n')
        fp.close()

        fp = open(os.path.join(save_dir, 'train_ap.txt'), 'a')
        fp.write(conf['loss_type']+str(i)+'\n[')
        for train_score in train_APs_clean:
            fp.write('{:.4f},'.format(train_score))
        fp.write(']\n[')
        for train_score in val_APs_adv:
            fp.write('{:.4f},'.format(train_score))
        fp.write(']\n')

        fp.close()

    # if save_dir is not None:
    #     torch.save({'model':model.state_dict(), 'epoch':conf['epochs']}, os.path.join(save_dir, str(i) + '_last.ckpt'))



def train_classification(model, optimizer, train_loader, criterion=None, loss_type=None, epoch = 0):
    model.train()

    losses = []
    for i, (index, inputs, target) in enumerate(train_loader):

        optimizer.zero_grad()
        inputs = inputs.to(device)
        target = target.to(device).float()

        if loss_type == 'ce':
            out = torch.sigmoid(model(inputs))
            if len(target.shape) != 2:
                target = torch.reshape(target, (-1, 1))
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        elif loss_type in ['ap']:
            out = torch.sigmoid(model(inputs))
            loss = criterion(out, target, index)
            loss.backward()
            optimizer.step()
        elif loss_type in ['mm_ap']:
            adv_imgs = AP_attack(model, inputs, target, index, criterion.attack_fn, steps=6, step_size=0.01, eps=0.031)
            
            out_adv = torch.sigmoid(model(adv_imgs))
            loss = criterion(out_adv, target, index)
            loss.backward()
            optimizer.step()            
        elif loss_type in [ 'adv_2', 'adv_x']:           
            adv_imgs = PGD_attack(model, inputs, torch.reshape(target, (-1, 1)), steps=6, step_size=0.01, eps=0.031)
            out = model(inputs)
            out_adv = model(adv_imgs)

            loss = criterion(out, out_adv, target, index)
            loss.backward()
            optimizer.step()
        elif loss_type in [ 'adv_2m']:           
            adv_imgs = ranking_attack_v1(model, inputs, index,torch.reshape(target, (-1, 1)), criterion.attack_fn, steps=6, step_size=0.01, eps=0.031)#steps=10, step_size=0.007, eps=0.031
            out = model(inputs)
            out_adv = model(adv_imgs)

            loss = criterion(out, out_adv, target, index)
            loss.backward()
            optimizer.step()
        elif loss_type in ['tra']:
            if len(target.shape) != 2:
                target = torch.reshape(target, (-1, 1))
            adv_imgs = TRADES_attack(model, inputs, target, steps=6, step_size=0.01, eps=0.031)
            out = torch.sigmoid(model(inputs))
            out_adv = torch.sigmoid(model(adv_imgs))

            loss = criterion(out, out_adv, target)
            loss.backward()
            optimizer.step()
        elif loss_type in ['adv_t']:
            if len(target.shape) != 2:
                target = torch.reshape(target, (-1, 1))
            adv_imgs = TRADES_attack(model, inputs, target, steps=6, step_size=0.01, eps=0.031)
            out = torch.sigmoid(model(inputs))
            out_adv = torch.sigmoid(model(adv_imgs))

            loss = criterion(out, out_adv, target, index)
            loss.backward()
            optimizer.step()            
        elif loss_type in ['mart']:
            if len(target.shape) != 2:
                target = torch.reshape(target, (-1, 1))
            adv_imgs = PGD_attack(model, inputs, target, steps=6, step_size=0.01, eps=0.031)
            out = torch.sigmoid(model(inputs))
            out_adv = torch.sigmoid(model(adv_imgs))

            loss = criterion(out, out_adv, target)
            loss.backward()
            optimizer.step()
            
        elif loss_type in ['pgd']:
            if len(target.shape) != 2:
                target = torch.reshape(target, (-1, 1))
            adv_imgs = PGD_attack(model, inputs, target, steps=6, step_size=0.01, eps=0.031)
            out_adv = torch.sigmoid(model(adv_imgs))

            loss = criterion(out_adv, target)
            loss.backward()
            optimizer.step()
             
        
        losses.append(loss.item())

    
    return sum(losses)/ len(losses)

def valTrain_classification(model, valTrain_loader):
    model.eval()

    #####evaluate clean ap
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for (index, inputs, target)in valTrain_loader:

        inputs = inputs.to(device)
        target = target.to(device).float()
        if len(target.shape) != 2:
            target = torch.reshape(target, (-1, 1))

        with torch.no_grad():
            pred = torch.sigmoid(model(inputs))

        preds = torch.cat([preds, pred], dim=0)
        targets = torch.cat([targets, target], dim=0)

    clean_ap = average_precision_score(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())

    return clean_ap


def val_classification(model, test_loader):
    model.eval()

    #####evaluate clean and adversarial ap
    preds = torch.Tensor([]).to(device)
    adv_preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for (index, inputs, target)in test_loader:
        inputs = inputs.to(device)
        target = target.to(device).float()
        if len(target.shape) != 2:
            target = torch.reshape(target, (-1, 1))

        adv_imgs = PGD_attack(model, inputs, target,  
                             steps=10, step_size=0.007, eps=0.031)

        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(inputs))
            adv_pred = torch.sigmoid(model(adv_imgs))

        preds = torch.cat([preds, pred], dim=0)
        adv_preds = torch.cat([adv_preds, adv_pred], dim=0)
        targets = torch.cat([targets, target], dim=0)


    clean_ap = average_precision_score(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())
    adv_ap = average_precision_score(targets.cpu().detach().numpy(), adv_preds.cpu().detach().numpy())

    return [clean_ap, adv_ap]
