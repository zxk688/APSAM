#这是一个seganythng的demo
#先导入必要的包
import random
import os
import shutil
import sys
import argparse
import re
from tqdm  import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # 或者使用其他的backend，如'Qt5Agg'、'Agg','tkAgg'等
from PIL import Image, ImageDraw
from Validation import val_multi
from torchvision import transforms
from datetime import datetime

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录及其所有内容
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry
import logging

transform = transforms.Compose([
    transforms.ToTensor(),])
# 配置日志文件
logging.basicConfig(filename='log.txt', level=logging.INFO)

# 重定向print输出到日志文件
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    logging.info(*args, **kwargs)

# 使用log_print代替print
def saveprompt(pth, img,  countors, points = None, pointslabel = None, boxes = None):
    if points != None:
        points = torch.squeeze(points, dim = 1)
        pointslabel = torch.squeeze(pointslabel, dim = 1)
    # img = img.cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))
    contour_image = cv2.drawContours(img, countors, -1, (0, 255, 0), 2)
    # 创建绘图对象
    # if contour_image.shape[2] == 3:  # 确保图像是彩色的
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB) # BGR
    
    contour_image = Image.fromarray(contour_image) # PIL.ImageDraw.Draw 只能处理 PIL.Image 对象
    draw = ImageDraw.Draw(contour_image)

    if points is not None:
        for i, point in enumerate(points):
            if pointslabel[i] == 1:
                # 标记绿点
                draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill=(0, 0, 255))
            else:
                # 标记红点
                draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill=(255, 0, 0))

    if boxes is not None:
        for box in boxes:
            # 提取坐标
            x1, y1, x2, y2 = box.tolist()  # 将张量转换为 Python 列表
            # 绘制矩形框
            draw.rectangle((x1, y1, x2, y2), outline=(228,0,127), width=3)

    # 保存图片
    contour_image.save(pth)


def find_contours_colors(cam_image, index):
    w, h = cam_image.shape[0], cam_image.shape[1]

    # 固定阈值
    lower_color = 120
    upper_color = 255
    binary_image = cv2.inRange(cam_image, lower_color, upper_color)
    # 自适应Outs阈值
    # otsu_threshold, binary_image = cv2.threshold(cam_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    os.makedirs("./output", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(output_path, binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_repaired = []
    target_boxes = []
    min_contour_area = 200  # 最小轮廓面积

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_contour_area:
            continue
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 5:
            continue
        contours_repaired.append(approx)
        target_boxes.append(np.array([x, y, x + w, y + h], dtype=np.int32))

    b_no_point, point_coords, point_labels = calc_point_coords(contours_repaired)
    return b_no_point, point_coords, point_labels, target_boxes, contours_repaired

def calc_point_coords(contours):
    point_coords = []
    point_labels = []
    b_no_point =True
    
    for i in range(len(contours)):
        moments_i = cv2.moments(contours[i])
        if  (moments_i['m00']) == 0:
            continue
        else:
            cx_i = int(moments_i['m10'] / moments_i['m00'])
            cy_i = int(moments_i['m01'] / moments_i['m00'])
        point_coords.append([cx_i, cy_i])
        point_labels.append(1)
        b_no_point =False

    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    return b_no_point, point_coords, point_labels

def save_mask(mask, save_path_file):
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(save_path_file, mask_uint8)

def get_all_target_bboxes(gray_img):
    ret, thresh = cv2.threshold(gray_img, 2, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        target_boxes.append((x, y, x+w, y+h))

    return target_boxes

def main():
    print = log_print
    #官方demo加载模型的方式
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_path = r'../z-sam_pretrained/sam_vit_b_01ec64.pth'
    # model_path = r'../z-sam_pretrained/sam_vit_l_0b3195.pth'
    model_path = r'../z-sam_pretrained/sam_vit_h_4b8939.pth'

    sam = sam_model_registry["vit_h"](checkpoint=model_path)    # 注意不同模型有vit_b vit_l
    sam = sam.to(device)
    predictor = SamPredictor(sam)

#turkey
    # image_path=r"./dataset/Turkey/train/sst2/"
    # label_path=r"./dataset/Turkey/train/label/"
    # cam_path = r"./result_Turkey/cam/"
    # output_prompt_path = r"./result_Turkey/prompt_result/"
    # output_path = r"./result/TK_pseudo_label/"

# HongKong
    image_path=r"./dataset/HongKong/train/sst2/"
    label_path=r"./dataset/HongKong/train/label/"
    cam_path = r"./result_HongKong/cam/"
    output_prompt_path = r"./result_HongKong/prompt_result/"
    output_path = r"./dataset/HongKong/train/pseudo_label/"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_prompt_path, exist_ok=True)
    for file_name in tqdm(os.listdir(image_path)):
        b_no_point = True
        i = re.search(r'\d+', file_name).group()
        # i = 89
        image = cv2.imread(image_path + str(i)+".png")
        label = cv2.imread(label_path + str(i)+".png")
        pred = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        pred_path = os.path.join(output_path, f"{i}.png")

        # 有无热力图区域
        if cv2.imread(cam_path + str(i) + ".png") is None:
            # save_mask(pred, os.path.join(output_path1, f"{i}.png"))
            save_mask(pred, pred_path)
            continue
        else:
            cam_image = cv2.imread(cam_path + str(i) + ".png", cv2.IMREAD_GRAYSCALE)

        b_no_point, point_coords, point_labels, box, contours = find_contours_colors(cam_image, str(i))

        box = np.array(box)
        box = torch.from_numpy(box)
        point_coords = torch.from_numpy(np.array(point_coords))
        point_labels = torch.from_numpy(np.array(point_labels))
        point_coords = torch.unsqueeze(point_coords,dim=1)
        point_labels = torch.unsqueeze(point_labels,dim=1)

        if b_no_point == True: # 如何现有方法找不到热力点，使用辅助cam图
            save_mask(pred, pred_path)
            continue
        
        input_boxes = predictor.transform.apply_boxes_torch(box, image.shape[:2])
        input_points = predictor.transform.apply_coords_torch(point_coords, image.shape[:2])

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            pmasks, _, _ = predictor.predict_torch(point_coords=input_points.to(device), point_labels=point_labels.to(device), boxes=input_boxes.to(device), mask_input=None, multimask_output=True, return_logits=False) #点框
            # pmasks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=input_boxes.to(device), mask_input=None, multimask_output=True, return_logits=False)   #框
            # pmasks, _, _ = predictor.predict_torch(point_coords=input_points.to(device), point_labels=point_labels.to(device), boxes=None, mask_input=None, multimask_output=True, return_logits=False)   #点

        pred = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask1 in pmasks:
            mask1 = mask1.cpu().numpy()
            mask1 = mask1.astype(np.uint8)[0,:,:]
            pred = pred + mask1

        # 保存热力图/图+框+点
        saveprompt(os.path.join(output_prompt_path + str(i) + "imgprompt" + ".png"), image, contours, point_coords, point_labels,box)
        saveprompt(os.path.join(output_prompt_path + str(i) + "label" + ".png"), label, contours, point_coords, point_labels,box)
        save_mask(pred, os.path.join(output_prompt_path + str(i) + "result" + ".png"))
        save_mask(pred, pred_path)        

    [Accu,Precision,Recall,F1,kappa,IoU,Specificity,Sensitivity]=val_multi(output_path,label_path,2)
    return [Accu,Precision,Recall,F1,kappa,IoU,Specificity,Sensitivity]

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()