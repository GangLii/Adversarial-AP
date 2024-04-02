import torch
import numpy as np
from torch.utils.data.sampler import Sampler
import torch.nn as nn

class DataSampler(Sampler):
    def __init__(self, labels, batch_size, pos_num=1):
        r"""Arguments:
            labels (list or numpy.array): labels of training dataset, the shape of labels should be (n,)
            batch_size (int): how many samples per batch to load
            pos_num (int): specify how many positive samples in each batch
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_num, self.neg_num = pos_num, self.batch_size - pos_num

        self.posList = np.flatnonzero(self.labels==1)
        self.negList = np.flatnonzero(self.labels==0)
        self.posLen, self.negLen = len(self.posList), len(self.negList)
        self.posPtr, self.negPtr = 0, 0

        np.random.shuffle(self.posList)
        np.random.shuffle(self.negList)

        #### define how many batchs per epoch
        # self.batchNum = max(self.posLen//self.pos_num, self.negLen//self.neg_num)
        self.batchNum = len(self.labels)//self.batch_size
        self.ret = np.empty(self.batchNum*self.batch_size, dtype=np.int64)

    def __iter__(self):
        r""" This functuin will return a new Iterator object for an epoch."""

        for batch_id in range(self.batchNum):
            #### load postive samples
            beg = batch_id*self.batch_size
            if self.posPtr+self.pos_num >= self.posLen:
                temp = self.posList[self.posPtr:]
                np.random.shuffle(self.posList)
                self.posPtr = (self.posPtr+self.pos_num)%self.posLen
                self.ret[beg:beg+self.pos_num]= np.concatenate((temp,self.posList[:self.posPtr]))
            else:
                self.ret[beg:beg+self.pos_num]= self.posList[self.posPtr: self.posPtr+self.pos_num]
                self.posPtr += self.pos_num

            ### load negative samples
            beg += self.pos_num
            if self.negPtr+self.neg_num >= self.negLen:
                temp = self.negList[self.negPtr:]
                np.random.shuffle(self.negList)
                self.negPtr = (self.negPtr+self.neg_num)%self.negLen
                self.ret[beg:beg+self.neg_num]= np.concatenate((temp,self.negList[:self.negPtr]))
            else:
                self.ret[beg:beg+self.neg_num]= self.negList[self.negPtr: self.negPtr+self.neg_num]
                self.negPtr += self.neg_num

        return iter(self.ret)


    def __len__ (self):
        return len(self.ret)

###This is the objective of AdAP_LN and AdAP_LZ in the paper.
class AdAP_LN(nn.Module):  
    def __init__(self, margin, gamma, Lambda, data_length, loss_type = 'sqh'):
        super(AdAP_LN, self).__init__()
        self.u_all = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.u_pos = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.u_r = torch.zeros(3).cuda()
        self.u_d = torch.zeros(3).cuda()
        self.margin = margin
        self.gamma1, self.gamma2 =gamma
        self.Lambda = Lambda
        self.loss_type = loss_type
        self.tau = 1

    def attack_fn(self, f_x, f_px, step):
        f_x = f_x/self.tau
        f_px = f_px/self.tau
        
        g2 = torch.exp(f_x).mean()
        g3 = torch.exp(f_px).mean()

        self.u_d[1]= (1 - self.gamma2) * self.u_r[1] + self.gamma2 *g2.detach()
        self.u_d[2]= (1 - self.gamma2) * self.u_r[2] + self.gamma2 *g3.detach()
            
        p1 = (-1/self.u_d[1] * torch.exp(f_x)).detach()
        p3 = (1/self.u_d[2] * torch.exp(f_px)).detach()

        att_loss = (p1*f_px).mean() + (p3*f_px).mean()

        return att_loss

    def forward(self,y_pred, y_pred_adv, y_true, index_s):

        #### TODO
        y_prob = torch.sigmoid(y_pred)  
        y_pred = torch.tanh(y_pred)/self.tau
        y_pred_adv = torch.tanh(y_pred_adv)/self.tau
        #####****compute natural loss****####
        ### Sometimes the shape of y_true could be (B,1), so we squeeze y_true to be (B,).
        y_true = y_true.squeeze()
        f_ps = y_prob[y_true == 1].reshape(-1,1)
        index_ps= index_s[y_true == 1].reshape(-1)

        mat_data = y_prob.reshape(-1).repeat(len(f_ps), 1)
        pos_mask = (y_true == 1).reshape(-1)

        sur_loss = torch.max(self.margin - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2
        pos_sur_loss = sur_loss * pos_mask

        ### moving average
        self.u_all[index_ps] = (1 - self.gamma1) * self.u_all[index_ps] + self.gamma1 * (sur_loss.mean(1, keepdim=True).detach())
        self.u_pos[index_ps] = (1 - self.gamma1) * self.u_pos[index_ps] + self.gamma1 * (pos_sur_loss.mean(1, keepdim=True).detach())

        ###size of p: len(f_ps)* len(y_prob)
        p = (self.u_pos[index_ps] - (self.u_all[index_ps]) * pos_mask) / (self.u_all[index_ps] ** 2)

        p.detach_()
        nat_loss = torch.mean(p * sur_loss)

        #####****compute advsarial loss****####
        #### TODO print f_x,f_px to see if we need to clip the scores
        g1 = (torch.exp(y_pred)*(y_pred - y_pred_adv)).mean()
        g2 = torch.exp(y_pred).mean()
        g3 = torch.exp(y_pred_adv).mean()
        #### update estimators
        self.u_r[0]= (1 - self.gamma2) * self.u_r[0] + self.gamma2 *g1.detach()
        self.u_r[1]= (1 - self.gamma2) * self.u_r[1] + self.gamma2 *g2.detach()
        self.u_r[2]= (1 - self.gamma2) * self.u_r[2] + self.gamma2 *g3.detach()


        adv_loss = 1/self.u_r[1]  * g1    \
                    - self.u_r[0]/(self.u_r[1]**2) * g2  \
                    + 1/self.u_r[2] * g3  \
                    - 1/self.u_r[1] * g2


        loss = nat_loss + self.Lambda * adv_loss

        return loss

##alias
AdAP_LZ = AdAP_LN

###This is AdAP_LPN in the paper.
class AdAP_LPN(nn.Module):  
    def __init__(self, margin, gamma, Lambda, data_length, loss_type = 'sqh'):
        super(AdAP_LPN, self).__init__()
        self.u_all = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.u_pos = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.u_r = torch.zeros(3).cuda()
        self.u_d = torch.zeros(3).cuda()
        self.margin = margin
        self.gamma1, self.gamma2 =gamma
        self.Lambda = Lambda
        self.loss_type = loss_type
        ### Horizontal robustness
        self.kl_fn = nn.KLDivLoss(reduction='batchmean') 
        self.eps = 1e-12


    def forward(self,y_pred, y_pred_adv, y_true, index_s):

        #### TODO
        y_prob = torch.sigmoid(y_pred)  
        y_prob_adv = torch.sigmoid(y_pred_adv)
        # y_prob = torch.sigmoid(y_pred_adv)  ##adv
        y_pred = torch.tanh(y_pred)
        y_pred_adv = torch.tanh(y_pred_adv)
        #####****compute natural loss****####
        ### Sometimes the shape of y_true could be (B,1), so we squeeze y_true to be (B,).
        y_true = y_true.squeeze()
        f_ps = y_prob[y_true == 1].reshape(-1,1)
        index_ps= index_s[y_true == 1].reshape(-1)

        mat_data = y_prob.reshape(-1).repeat(len(f_ps), 1)
        pos_mask = (y_true == 1).reshape(-1)

        sur_loss = torch.max(self.margin - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2
        pos_sur_loss = sur_loss * pos_mask

        ### moving average
        self.u_all[index_ps] = (1 - self.gamma1) * self.u_all[index_ps] + self.gamma1 * (sur_loss.mean(1, keepdim=True).detach())
        self.u_pos[index_ps] = (1 - self.gamma1) * self.u_pos[index_ps] + self.gamma1 * (pos_sur_loss.mean(1, keepdim=True).detach())

        ###size of p: len(f_ps)* len(y_prob)
        p = (self.u_pos[index_ps] - (self.u_all[index_ps]) * pos_mask) / (self.u_all[index_ps] ** 2)

        p.detach_()
        nat_loss = torch.mean(p * sur_loss)

        #####****compute advsarial loss****####
        #### TODO print f_x,f_px to see if we need to clip the scores
        g1 = (torch.exp(y_pred)*(y_pred - y_pred_adv)).mean()
        g2 = torch.exp(y_pred).mean()
        g3 = torch.exp(y_pred_adv).mean()
        #### update estimators
        self.u_r[0]= (1 - self.gamma2) * self.u_r[0] + self.gamma2 *g1.detach()
        self.u_r[1]= (1 - self.gamma2) * self.u_r[1] + self.gamma2 *g2.detach()
        self.u_r[2]= (1 - self.gamma2) * self.u_r[2] + self.gamma2 *g3.detach()

        adv_loss_v = 1/self.u_r[1]  * g1    \
                    - self.u_r[0]/(self.u_r[1]**2) * g2  \
                    + 1/self.u_r[2] * g3  \
                    - 1/self.u_r[1] * g2
        ###Horizontal robustness
        ### expand to two classes
        y_dist = torch.cat([y_prob,1-y_prob],dim=-1)
        y_dist_adv = torch.cat([y_prob_adv,1-y_prob_adv],dim=-1)
        adv_loss_h = self.kl_fn(torch.log(y_dist_adv + self.eps), y_dist)

        adv_loss = adv_loss_v + adv_loss_h
        
        loss = nat_loss + self.Lambda * adv_loss

        return loss
    
###This is AP Max. in the paper.    
class AUPRC_Loss(torch.nn.Module): 
    def __init__(self, margin=1.0, gamma=0.1, data_length=None, device=None):
        """Arguments:
            data_length:the number of samples in training dataset
            margin (float): margin for squred hinge loss, e.g., m in [0, 1]
            gamma (float): factor for moving average
        """
        super(AUPRC_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device      

        self.u_all = torch.tensor([0.0]*data_length).view(-1, 1).to(self.device)
        self.u_pos = torch.tensor([0.0]*data_length).view(-1, 1).to(self.device)
        self.margin = margin
        self.gamma = gamma
        print('self.margin',self.margin,'self.gamma',self.gamma)
        
    def forward(self, y_pred, y_true, index_s): 
        """Arguments:
            y_pred : prediction scores of batched samples
            y_true : ground truth of batched samples
            index_s : batched samples' indices in dataset
        Returns:
            loss
        """     
        ### Sometimes the shape of y_true could be (B,1), so we squeeze y_true to be (B,).
        y_true = y_true.squeeze()
        f_ps = y_pred[y_true == 1].reshape(-1,1)
        index_ps = index_s[y_true == 1].reshape(-1)
  
        mat_data = y_pred.reshape(-1).repeat(len(f_ps), 1)
        pos_mask = (y_true == 1).reshape(-1)

        sur_loss = torch.max(self.margin - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2
        pos_sur_loss = sur_loss * pos_mask

        ### moving average
        self.u_all[index_ps] = (1 - self.gamma) * self.u_all[index_ps] + self.gamma * (sur_loss.mean(1, keepdim=True).detach())
        self.u_pos[index_ps] = (1 - self.gamma) * self.u_pos[index_ps] + self.gamma * (pos_sur_loss.mean(1, keepdim=True).detach())

        ###size of p: len(f_ps)* len(y_pred)
        p = (self.u_pos[index_ps] - (self.u_all[index_ps]) * pos_mask) / (self.u_all[index_ps] ** 2)
        p.detach_()
        loss = torch.mean(p * sur_loss)
        
        return loss


###This is AdAP_MM in the paper. 
class AdAP_MM(nn.Module):  
    def __init__(self, margin, gamma, data_length):
        super(AdAP_MM, self).__init__()
        self.u_all = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.u_pos = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.margin = margin
        self.gamma = gamma

    def attack_fn(self, f_px, y_true, index_s):
        y_true = y_true.squeeze()
        f_ps = f_px[y_true == 1].reshape(-1,1)
        index_ps = index_s[y_true == 1].reshape(-1)
  
        mat_data = f_px.reshape(-1).repeat(len(f_ps), 1)
        pos_mask = (y_true == 1).reshape(-1)

        sur_loss = torch.max(self.margin - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2
        pos_sur_loss = sur_loss * pos_mask

        ### moving average
        u_all = (1 - self.gamma) * self.u_all[index_ps] + self.gamma * (sur_loss.mean(1, keepdim=True).detach())
        u_pos = (1 - self.gamma) * self.u_pos[index_ps] + self.gamma * (pos_sur_loss.mean(1, keepdim=True).detach())

        ###size of p: len(f_ps)* len(y_pred)
        p = (u_pos - u_all * pos_mask) / (u_all ** 2)
        p.detach_()
        att_loss = torch.mean(p * sur_loss)

        return att_loss

    def forward(self, y_pred_adv, y_true, index_s): 
        """Arguments:
            y_pred : prediction scores of batched samples
            y_true : ground truth of batched samples
            index_s : batched samples' indices in dataset
        Returns:
            loss
        """     
        ### Sometimes the shape of y_true could be (B,1), so we squeeze y_true to be (B,).
        y_true = y_true.squeeze()
        f_ps = y_pred_adv[y_true == 1].reshape(-1,1)
        index_ps = index_s[y_true == 1].reshape(-1)
  
        mat_data = y_pred_adv.reshape(-1).repeat(len(f_ps), 1)
        pos_mask = (y_true == 1).reshape(-1)

        sur_loss = torch.max(self.margin - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2
        pos_sur_loss = sur_loss * pos_mask

        ### moving average
        self.u_all[index_ps] = (1 - self.gamma) * self.u_all[index_ps] + self.gamma * (sur_loss.mean(1, keepdim=True).detach())
        self.u_pos[index_ps] = (1 - self.gamma) * self.u_pos[index_ps] + self.gamma * (pos_sur_loss.mean(1, keepdim=True).detach())

        ###size of p: len(f_ps)* len(y_pred)
        p = (self.u_pos[index_ps] - (self.u_all[index_ps]) * pos_mask) / (self.u_all[index_ps] ** 2)
        p.detach_()
        loss = torch.mean(p * sur_loss)
        
        return loss
   
###This is AdAP_PZ in the paper. 
class AdAP_PZ(nn.Module): 
    def __init__(self, margin, gamma, Lambda, data_length):
        super(AdAP_PZ, self).__init__()
        self.u_all = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.u_pos = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.margin = margin
        self.gamma =gamma
        self.Lambda = Lambda
        ### TRADES robustness
        self.robust_fn = nn.KLDivLoss(reduction='batchmean') 
        self.eps = 1e-12


    def forward(self,y_pred, y_pred_adv, y_true, index_s):

        #####****compute natural loss****####
        ### Sometimes the shape of y_true could be (B,1), so we squeeze y_true to be (B,).
        y_true = y_true.squeeze()
        f_ps = y_pred[y_true == 1].reshape(-1,1)
        index_ps= index_s[y_true == 1].reshape(-1)

        mat_data = y_pred.reshape(-1).repeat(len(f_ps), 1)
        pos_mask = (y_true == 1).reshape(-1)

        sur_loss = torch.max(self.margin - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2
        pos_sur_loss = sur_loss * pos_mask

        ### moving average
        self.u_all[index_ps] = (1 - self.gamma) * self.u_all[index_ps] + self.gamma * (sur_loss.mean(1, keepdim=True).detach())
        self.u_pos[index_ps] = (1 - self.gamma) * self.u_pos[index_ps] + self.gamma * (pos_sur_loss.mean(1, keepdim=True).detach())

        ###size of p: len(f_ps)* len(y_prob)
        p = (self.u_pos[index_ps] - (self.u_all[index_ps]) * pos_mask) / (self.u_all[index_ps] ** 2)

        p.detach_()
        nat_loss = torch.mean(p * sur_loss)

        #####****compute advsarial loss****####
        ### expand to two classes
        y_prob = torch.cat([y_pred,1-y_pred],dim=-1)
        y_prob_adv = torch.cat([y_pred_adv,1-y_pred_adv],dim=-1)
        adv_loss = self.robust_fn(torch.log(y_prob_adv + self.eps), y_prob)


        loss = nat_loss + self.Lambda * adv_loss

        return loss
  
class TRADES(nn.Module):
    def __init__(self, Lambda=6):
        super(TRADES, self).__init__()
        self.robust_fn = nn.KLDivLoss(reduction='batchmean') 
        self.natural_fn = nn.BCELoss()
        self.Lambda = Lambda
        self.eps = 1e-12
        print('self.Lambda',self.Lambda)

    def forward(self,y_pred, y_pred_adv, y_true):

        clean_loss = self.natural_fn(y_pred, y_true)

        ### expand to two classes
        y_prob = torch.cat([y_pred,1-y_pred],dim=-1)
        y_prob_adv = torch.cat([y_pred_adv,1-y_pred_adv],dim=-1)
        adv_loss = self.robust_fn(torch.log(y_prob_adv + self.eps), y_prob)

        loss = clean_loss + self.Lambda * adv_loss

        return loss

class MART(nn.Module):
    def __init__(self, Lambda=6):
        super(MART, self).__init__()
        self.robust_fn = nn.KLDivLoss(reduction='none') 
        self.adv_fn = nn.BCELoss()
        self.Lambda = Lambda
        self.eps = 1e-12
        print('self.Lambda',self.Lambda)

    def forward(self,y_pred, y_pred_adv, y_true):

        adv_loss = self.adv_fn(y_pred_adv, y_true)

        ### expand to two classes
        y_prob = torch.cat([y_pred,1-y_pred],dim=-1)
        y_prob_adv = torch.cat([y_pred_adv,1-y_pred_adv],dim=-1)
        rob_loss = torch.sum( self.robust_fn(torch.log(y_prob_adv+self.eps), y_prob), dim =1, keepdim=True ) \
                    * (torch.abs(y_true - y_pred))
        rob_loss = rob_loss.mean()            

        loss = adv_loss + self.Lambda * rob_loss

        return loss
    
