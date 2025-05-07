import os
import torch
import random
import numpy as np
import argparse
import importlib
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from tensorboardX import SummaryWriter

from tool import pyutils, torchutils, visualization
from dataloader_cd import CAMDataset

cudnn.enabled = True

def setup_seed(seed):
    print("random seed is set to", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--network_name", default="network.resnet50_SIPE", type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-4, type=float)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--tf_freq", default=500, type=int)
    parser.add_argument("--val_freq", default=500, type=int)
    parser.add_argument("--dataset", default="HongKong", type=str)
    parser.add_argument("--dataset_root", default="./dataset/HongKong/", type=str)
    parser.add_argument("--session_name", default="result_HongKong", type=str)
    parser.add_argument("--seed", default=15, type=int)
    args = parser.parse_args()

    setup_seed(args.seed)

    os.makedirs(args.session_name, exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'snapshot'), exist_ok=True)

    pyutils.Logger(os.path.join(args.session_name, args.session_name + '.log'))
    tblogger = SummaryWriter(os.path.join(args.session_name, 'runs'))

    model = getattr(importlib.import_module(args.network_name), 'Net')(num_cls=3)
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    train_dataset = CAMDataset(path_root=args.dataset_root, mode="train")
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizerSGD([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model = model.cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    for ep in range(args.max_epoches):
        print('Epoch %d/%d' % (ep + 1, args.max_epoches))

        for step, pack in enumerate(train_data_loader):
            img = pack['img'].cuda()
            n, c, h, w = img.shape
            label = pack['label'].cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)
            valid_mask = pack['valid_mask'].cuda()

            valid_mask[:, 1:] = valid_mask[:, 1:] * label
            valid_mask_lowres = F.interpolate(valid_mask, size=(h // 16, w // 16), mode='nearest')

            outputs = model.forward(img, valid_mask_lowres)
            score = outputs['score']
            norm_cam = outputs['cam']
            IS_cam = outputs['IS_cam']

            lossCLS = F.multilabel_soft_margin_loss(score, label)

            IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
            lossGSC = torch.mean(torch.abs(norm_cam - IS_cam))

            losses = lossCLS + lossGSC
            avg_meter.add({'lossCLS': lossCLS.item(), 'lossGSC': lossGSC.item()})

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if (optimizer.global_step - 1) % args.print_freq == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'lossCLS:%.4f' % (avg_meter.pop('lossCLS')),
                      'lossGSC:%.4f' % (avg_meter.pop('lossGSC')),
                      'imps:%.1f' % ((step + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_est_finish()), flush=True)

                tblogger.add_scalar('lossCLS', lossCLS, optimizer.global_step)
                tblogger.add_scalar('lossGSC', lossGSC, optimizer.global_step)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], optimizer.global_step)

            if (optimizer.global_step - 1) % args.tf_freq == 0:
                img_8 = visualization.convert_to_tf(img[0])
                norm_cam = F.interpolate(norm_cam, img_8.shape[1:], mode='bilinear')[0].detach().cpu().numpy()
                IS_cam = F.interpolate(IS_cam, img_8.shape[1:], mode='bilinear')[0].detach().cpu().numpy()

                CAM = visualization.generate_vis(norm_cam, None, img_8,
                                                 func_label2color=visualization.VOClabel2colormap,
                                                 threshold=None, norm=False)
                IS_CAM = visualization.generate_vis(IS_cam, None, img_8,
                                                    func_label2color=visualization.VOClabel2colormap,
                                                    threshold=None, norm=False)

                tblogger.add_images('CAM', CAM, optimizer.global_step)
                tblogger.add_images('IS_CAM', IS_CAM, optimizer.global_step)

        timer.reset_stage()

    torch.save({'net': model.state_dict()}, os.path.join(args.session_name, 'snapshot/final_cd.pth'))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
