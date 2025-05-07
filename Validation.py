from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from torch.utils import data
# import dataloader

def val_sing(test_path,label_path):



    res=np.array(Image.open(test_path)).astype(np.int64)
    ref=np.array(Image.open(label_path)).astype(np.int64)

    res[res==255]=1
    ref[ref==255]=1

    TP = np.sum(res*ref==1)
    FN = np.sum(ref*(1-res)==1)
    FP = np.sum(res*(1-ref)==1)
    TN = np.sum((1-res)*(1-ref)==1)

    print(f'TP={TP} | TN={TN} | FP={FP} | FN={FN}')

    Accu=(TP+TN)/(TP+TN+FP+FN)
    Precision=(TP)/(TP+FP)
    Recall=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    Sensitivity = TP/(TP+FN)
    F1=2*((Precision*Recall)/(Precision+Recall))

    pe=((TP+FN)*(TP+FP)+(TN+FP)*(TN+FN))/((TP+TN+FP+FN)**2) 
    kappa=(Accu-pe)/(1-pe)
    IoU=TP/(TP+FP+FN)

    print(f'Accu={Accu} Precision={Precision} Recall={Recall} F1={F1} kappa={kappa} IoU={IoU} Specificity={Specificity} Sensitivity ={Sensitivity}')


def val_multi(test_path,label_path, decimal_places = 6):
    # decimal_places:Round the results to x decimal places

    TP , FN, FP, TN= 0, 0,0,0
    file_names = sorted(os.listdir(label_path))

    for file_name in tqdm(file_names):

        # res=np.array(Image.open(test_path+str(i+1)+".png")).astype(np.int64)
        # ref=np.array(Image.open(label_path+"test_"+str(i+1)+".png")).astype(np.int64)
        # res=np.array(Image.open(test_path+str(i)+".png")).astype(np.int64)
        # ref=np.array(Image.open(label_path+str(i)+".png")).astype(np.int64)
        res=np.array(Image.open(test_path+file_name)).astype(np.int64)
        ref=np.array(Image.open(label_path+file_name)).astype(np.int64)
        # ref=np.array(Image.open(label_path+"0_"+str(i)+".png")).astype(np.int64)
        
        res[res==255]=1
        ref[ref==255]=1
        # print(res.shape, ref.shape)
        if res.ndim == 3:
            res = res[:, :, 0]

        TP = TP+np.sum(res*ref==1)          # 白色为正例
        FN = FN+np.sum(ref*(1-res)==1)
        FP =FP+ np.sum(res*(1-ref)==1)
        TN = TN+np.sum((1-res)*(1-ref)==1)
    print(f'TP={TP} | TN={TN} | FP={FP} | FN={FN}')

    Accu=(TP+TN)/(TP+TN+FP+FN)
    Precision=(TP)/(TP+FP)
    Recall=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    Sensitivity = TP/(TP+FN)
    F1=2*((Precision*Recall)/(Precision+Recall))

    pe=((TP+FN)*(TP+FP)+(TN+FP)*(TN+FN))/((TP+TN+FP+FN)**2) 
    kappa=(Accu-pe)/(1-pe)
    IoU=TP/(TP+FP+FN)

    IoU_pos = TP / (TP + FP + FN)
    IoU_neg = TN / (TN + FP + FN)

    # 计算 mIoU
    mIoU = (IoU_pos + IoU_neg) / 2

    [Accu,Precision,Recall,F1,kappa,IoU,Specificity,Sensitivity] = [round(x*100, decimal_places) for x in [Accu,Precision,Recall,F1,kappa,IoU,Specificity,Sensitivity]]
    print(f'Accu={Accu} Precision={Precision} Recall={Recall} F1={F1} kappa={kappa} IoU={IoU} Specificity={Specificity} Sensitivity ={Sensitivity}')
    return [Accu,Precision,Recall,F1,kappa,IoU,Specificity,Sensitivity]
