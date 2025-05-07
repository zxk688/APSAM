import os
import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from torch.backends import cudnn
from PIL import Image
import matplotlib.pyplot as plt
import imageio

from tool import torchutils, pyutils, imutils
from dataloader_cd import CDDatasetMSF

cudnn.enabled = True


def overlap(img, hm):
    """将热力图叠加在原图上"""
    hm = plt.cm.jet(hm)[:, :, :3]
    hm = np.array(Image.fromarray((hm * 255).astype(np.uint8), 'RGB')
                  .resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(np.float32) * 2
    img = np.array(img).astype(np.float32)
    if hm.shape == img.shape:
        out = (hm + img) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
        return out
    else:
        print("Shape mismatch:", hm.shape, img.shape)
        return img.astype(np.uint8)


def draw_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
    """保存每个类别的可视化热力图"""
    gt_cat = np.where(gt_label == 1)[0]
    for gt in gt_cat:
        heatmap = overlap(orig_img, norm_cam[gt])
        save_name = f"{img_name}_{gt}.png"
        imageio.imsave(os.path.join(save_path, save_name), heatmap)


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    heatmap_output_dir = os.path.join(args.session_name, 'cam')
    os.makedirs(heatmap_output_dir, exist_ok=True)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        for pack in data_loader:
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            label = F.pad(label, (1, 0), 'constant', 1.0)  # 背景类别补1

            # 获取多尺度图像输入，并进行模型推理
            outputs = [model(img[0].cuda(non_blocking=True), 
                             label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)) 
                       for img in pack['img']]

            # CAM 归一化与插值
            NORM_CAM_list = [F.interpolate(torch.unsqueeze(output[0].cpu(), 1), size,
                                           mode='bilinear', align_corners=False)
                             for output in outputs]
            NORM_CAM = torch.sum(torch.stack(NORM_CAM_list, 0), 0)[:, 0]
            NORM_CAM /= F.adaptive_max_pool2d(NORM_CAM, (1, 1)) + 1e-5
            NORM_CAM = NORM_CAM.cpu().numpy()

            # 这里只保存第三个类别的 CAM（可根据需要调整）
            plt.imsave(os.path.join(heatmap_output_dir, f"{img_name}.png"), NORM_CAM[2], cmap='gray')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet50_SIPE", type=str)
    parser.add_argument("--num_workers", default=os.cpu_count() // 2, type=int)
    parser.add_argument("--session_name", default="result_HongKong", type=str)
    parser.add_argument("--ckpt", default="final_cd.pth", type=str)
    parser.add_argument("--dataset", default="HongKong", type=str)
    args = parser.parse_args()

    pyutils.Logger(os.path.join(args.session_name, 'infer.log'))

    # 加载模型
    model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=3)
    checkpoint = torch.load(os.path.join(args.session_name, 'snapshot', args.ckpt))
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()

    # 加载数据
    dataset = CDDatasetMSF(path_root=f"./dataset/{args.dataset}/", mode="train")
    dataset = torchutils.split_dataset(dataset, torch.cuda.device_count())

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=torch.cuda.device_count(), args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
