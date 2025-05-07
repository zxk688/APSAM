import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from dataloader_cd import Dataset
import time
from tensorboardX import SummaryWriter
import dataloader_cd as dataloader
from resunet import ResUnet
import numpy as np
from utils import *
from tqdm import tqdm 
import importlib

from network.resnet50 import resnet50


batch_size = 24
epoch = 100
base_lr = 1e-2
save_iter = 2

set_num_workers = 4
#set_momentum = 0.9
#set_weight_decay = 0.001
eval=True

# nohup python -u step4_train_resunet.py >tk_abla_sam2_t.log 2>&1 &
def loss_calc(pred,label):
    label = torch.squeeze(label,dim=1)
    pred = torch.squeeze(pred,dim=1)
    loss = nn.BCELoss()#
    return loss(pred,label)



def main():
    bestF1 = 0
    writer=SummaryWriter(comment="resunet on dataset")#
    # device = torch.device('cuda:0')#
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResUnet()
    model.to(device)
    set_snapshot_dir = f"./snapshot/"

    trainloader=data.DataLoader(
            Dataset(path_root="dataset/HongKong/" ,mode="train", pseudo_label = 'pseudo_label'),
            batch_size=batch_size,shuffle=True,num_workers=set_num_workers,pin_memory=True)
    if eval==True:
        evalloader=data.DataLoader(
                dataloader.Dataset(path_root="dataset/HongKong/",mode="test"),
                batch_size=1,shuffle=False,num_workers=set_num_workers,pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',verbose=True,patience=5,cooldown=3,min_lr=1e-8,factor=0.5)
    
    for i in range(epoch):
        torch.cuda.empty_cache()
        loss_list=[]
        model.train()

        for batch in enumerate((trainloader)):
            optimizer.zero_grad()
            img, label = batch[1]
            img = img.to(device)#
            # print(img.shape)
            label = label.to(device)
            pred = model(img)#
            # print(pred.shape)
            loss = loss_calc(pred,label)
            loss_list.append(loss.item())#
            loss.backward()#
            optimizer.step()#
        scheduler.step(sum(loss_list)/len(loss_list))#
        lr = optimizer.param_groups[0]['lr']
        print(time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))+f', epoch={i} | loss={sum(loss_list)/len(loss_list):.7f} | lr={lr:.7f}')#显示的也是平均loss
        writer.add_scalar('scalar/train_loss',sum(loss_list)/len(loss_list),i)#
        if (i+1)%save_iter==0 and i!=0:
            model.eval()

            all_preds = []
            all_gts = []
            for _,batch in enumerate((evalloader)):
                img, label = batch
                img = img.to(device)
                label = label.to(device)
                pred = model(img)
                
                zero = torch.zeros_like(pred)
                one = torch.ones_like(pred)
                pred = torch.where(pred > 0.5, one, pred)
                pred = torch.where(pred <= 0.5, zero, pred)
                # print(pred.shape)

                # pred = pred.detach().cpu().numpy().transpose((1, 2, 0))
                pred = pred.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
                all_preds.append(pred)
                all_gts.append(label.detach().cpu().numpy())
            accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                    np.concatenate([p.ravel() for p in all_gts]).ravel())
            if accuracy > bestF1:
                bestF1 = accuracy
                torch.save(model.state_dict(),set_snapshot_dir+time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))+"_resunet_"+str(i+1)+f"_{accuracy:.4f}"f"_{method}"+".pth")#在一定的epoch存储模型

            print(f'model saved at epoch{i+1},accuracy:{accuracy}')
            writer.add_scalar('scalar/eval_acc',accuracy,i)
            torch.cuda.empty_cache()

if __name__=="__main__":
    main()


    
    
