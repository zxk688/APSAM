from time import time
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from PIL import Image
import os
from torch.autograd import Variable
import cv2
import numpy as np
import cv2
from tqdm  import tqdm
from PIL import Image  
from Validation import val_multi
from resunet import ResUnet
import importlib


def main():
    #HK
    model_path="xxx.pth"
    test_path="./dataset/HongKong/train/sst2/"
    label_path="./dataset/HongKong/train/label/"
    output_path = "./result/HK/"
    #TK
    # model_path="./result_Turkey/snapshot/APSAM_boxpoint_ablation_0.7519.pth"
    # test_path="./dataset/Turkey/train/sst2/"
    # label_path="./dataset/Turkey/train/label/"
    # output_path = "./result/TK/"

    model=ResUnet() 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for i in tqdm(range(len(os.listdir(test_path)))):
        img=Image.open(test_path+str(i)+".png")
  
        img = tfs.ToTensor()(img).unsqueeze(0)
        img=img.to(device)
        pred=model(img)

        zero = torch.zeros_like(pred)
        one = torch.ones_like(pred)
        pred = torch.where(pred > 0.5, one, pred)
        pred = torch.where(pred <= 0.5, zero, pred)
        pred = pred.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
        cv2.imwrite(output_path+str(i)+".png", pred*255)
    [Accu,Precision,Recall,F1,kappa,IoU,Specificity,Sensitivity]=val_multi(output_path,label_path)
    return [Accu,Precision,Recall,F1,kappa,IoU,Specificity,Sensitivity]

if __name__=="__main__":
    main()


