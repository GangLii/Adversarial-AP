import torch
import torch.nn as nn
import torch.nn.functional as F



#### __imagenet_stats
__mean__ = torch.tensor([ [[0.485]],[[0.456]], [[0.406]] ]).cuda() 
__std__ =  torch.tensor([ [[0.229]],[[0.224]], [[0.225]] ]).cuda() 
def norm(img_tensors):
    return (img_tensors- __mean__)/ __std__

def de_norm(img_tensors):
    return img_tensors* __std__ + __mean__
       
    
def ranking_attack_v1(model, imgs, index, attack_fn=None, steps=10, step_size=0.007, eps=0.031):
    mode_backup = model.training
    model.eval()

    imgs = de_norm(imgs.detach())
    adv_imgs = imgs + 0.001 * torch.randn(imgs.size()).cuda() 
    for step in range(steps):
        adv_imgs.requires_grad_(True)
        with torch.enable_grad():
            pred_x = torch.tanh(model(norm(imgs)))
            pred_px = torch.tanh(model(norm(adv_imgs)))
            adv_loss = attack_fn(pred_x, pred_px, step)
        
        grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
        adv_imgs = adv_imgs + step_size* torch.sign(grad)
        adv_imgs = torch.min(torch.max(adv_imgs, imgs - eps), imgs + eps)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
    
    model.train(mode_backup)
    return norm(adv_imgs.detach()) # this detach may not be necessary


def AP_attack(model, imgs, targets, index, attack_fn=None, steps=10, step_size=0.007, eps=0.031):
    mode_backup = model.training
    model.eval()

    imgs = de_norm(imgs.detach())
    adv_imgs = imgs + 0.001 * torch.randn(imgs.size()).cuda() 
    for _ in range(steps):
        adv_imgs.requires_grad_(True)
        with torch.enable_grad():
            adv_loss = attack_fn(torch.sigmoid(model(norm(adv_imgs))), targets, index)
        
        grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
        adv_imgs = adv_imgs + step_size* torch.sign(grad)
        adv_imgs = torch.min(torch.max(adv_imgs, imgs - eps), imgs + eps)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
    
    model.train(mode_backup)
    return norm(adv_imgs.detach()) # this detach may not be necessary

def PGD_attack(model, imgs, targets, steps=10, step_size=0.007, eps=0.031):
    mode_backup = model.training
    model.eval()
    attack_fn = F.binary_cross_entropy

    imgs = de_norm(imgs.detach())
    adv_imgs = imgs + 0.001 * torch.randn(imgs.size()).cuda() 
    for _ in range(steps):
        adv_imgs.requires_grad_(True)
        with torch.enable_grad():
            adv_loss = attack_fn(torch.sigmoid(model(norm(adv_imgs))), targets)
        
        grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
        adv_imgs = adv_imgs + step_size* torch.sign(grad)
        adv_imgs = torch.min(torch.max(adv_imgs, imgs - eps), imgs + eps)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
    
    model.train(mode_backup)
    return norm(adv_imgs.detach()) # this detach may not be necessary

def TRADES_attack(model, imgs, targets, steps=10, step_size=0.007, eps=0.031):
    mode_backup = model.training
    model.eval()
    
    attack_fn = nn.KLDivLoss(reduction='batchmean') 
    imgs = de_norm(imgs.detach())
    adv_imgs = imgs + 0.001 * torch.randn(imgs.size()).cuda() 
    for _ in range(steps):
        adv_imgs.requires_grad_(True)
        with torch.enable_grad():
            pred_x = torch.sigmoid(model(norm(imgs)))
            pred_px = torch.sigmoid(model(norm(adv_imgs)))

            ### expand to two classes
            prob_x = torch.cat([pred_x,1-pred_x],dim=-1)
            prob_px = torch.cat([pred_px,1-pred_px],dim=-1)
            adv_loss = attack_fn(torch.log(prob_px), prob_x)
        
        grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
        adv_imgs = adv_imgs + step_size* torch.sign(grad)
        adv_imgs = torch.min(torch.max(adv_imgs, imgs - eps), imgs + eps)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
    
    model.train(mode_backup)
    return norm(adv_imgs.detach()) # this detach may not be necessary
