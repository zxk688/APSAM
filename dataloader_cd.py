
import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data
import imageio
from tool import pyutils, imutils, torchutils
import torchvision.transforms as tfs


IMG_FOLDER_NAME = "sst2"
MASK_FOLDER_NAME = "SegmentationClass"

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class Dataset(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train",pseudo_label = None):
        super(Dataset,self).__init__()
        self.path_root = path_root + "train"
        # self.rs_images_dir = os.listdir(os.path.join(self.path_root, "img"))
        # self.rs_images = [os.path.join(self.path_root, "img", img) for img in self.rs_images_dir]
        # self.gt_images_dir = os.listdir(os.path.join(self.path_root,"label"))
        # self.gt_images = [os.path.join(self.path_root,"label",img) for img in self.rs_images_dir]

        self.rs_images_dir = os.listdir(os.path.join(self.path_root, "sst2"))
        self.rs_images = [os.path.join(self.path_root, "sst2", img) for img in self.rs_images_dir]
        if mode == 'train':
            self.gt_images_dir = os.listdir(os.path.join(self.path_root, pseudo_label))
            self.gt_images = [os.path.join(self.path_root, pseudo_label, img) for img in self.rs_images_dir]
        else:
            self.gt_images_dir = os.listdir(os.path.join(self.path_root,"label"))
            self.gt_images = [os.path.join(self.path_root,"label",img) for img in self.rs_images_dir]


    def __getitem__(self, item):
        img = Image.open(self.rs_images[item])
        label = Image.open(self.gt_images[item])
        img = tfs.ToTensor()(img)
        label = tfs.ToTensor()(label)

        return img, label

    def __len__(self):
        return len(self.rs_images)
    
#label 是one-hot vector
class CAMDataset(data.Dataset):
    def __init__(self,path_root="./dataset/datasetLandslide/",mode=""):
        super(CAMDataset,self).__init__()
        self.path_root=path_root+mode
        self.sst2_images_dir = os.listdir(os.path.join(self.path_root, "sst2"))
        self.sst2_images = [os.path.join(self.path_root, "sst2", img) for img in self.sst2_images_dir]
        self.gt_images_dir=os.listdir(os.path.join(self.path_root,"label"))
        self.gt_images=[os.path.join(self.path_root,"label",img) for img in self.sst2_images_dir]
        self.img_normal = TorchvisionNormalize()

    def __getitem__(self, item):
        name = os.path.split(self.sst2_images[item])[-1]
        name = os.path.splitext(name)[0]
        
        img = np.asarray(Image.open(self.sst2_images[item]))
        gt = np.asarray(Image.open(self.gt_images[item]))

        img=self.img_normal(img)
        
        img = np.ascontiguousarray(imutils.HWC_to_CHW(img))
        
        gt_ = np.zeros_like(gt)
        gt_[gt==255] = 1

        label = torch.nn.functional.one_hot(torch.from_numpy(np.array((gt_).max())).to(torch.int64), num_classes=2)
        if np.unique(gt_)[0]==1:
            label[0] = 0
        else:
            label[0] = 1

        valid_mask = torch.ones((label.shape[0]+1, img.shape[1], img.shape[2]))

        return {'name': name, 'img': img, 'valid_mask': valid_mask, 'label': label}

    def __len__(self):
        return len(self.sst2_images)
    
#label 是one-hot vector
class CDDataset(data.Dataset):
    def __init__(self,path_root="./dataset/datasetLandslide/",mode=""):
        super(CDDataset,self).__init__()
        self.path_root=path_root+mode
        # self.sst1_images_dir=os.listdir(os.path.join(self.path_root,"sst1"))
        # self.sst1_images=[os.path.join(self.path_root,"sst1",img) for img in self.sst1_images_dir]

        self.sst2_images_dir = os.listdir(os.path.join(self.path_root, "sst2"))
        self.sst2_images = [os.path.join(self.path_root, "sst2", img) for img in self.sst2_images_dir]

        # self.sst2_images_dir = os.listdir(os.path.join(self.path_root, "img"))
        # self.sst2_images = [os.path.join(self.path_root, "img", img) for img in self.sst2_images_dir]
        self.gt_images_dir=os.listdir(os.path.join(self.path_root,"label"))
        self.gt_images=[os.path.join(self.path_root,"label",img) for img in self.sst2_images_dir]
        self.img_normal = TorchvisionNormalize()

    def __getitem__(self, item):
        name = os.path.split(self.sst2_images[item])[-1]
        name = os.path.splitext(name)[0]
        
        img = np.asarray(Image.open(self.sst2_images[item]))
        gt = np.asarray(Image.open(self.gt_images[item]))

        img=self.img_normal(img)
        
        img = np.ascontiguousarray(imutils.HWC_to_CHW(img))
        
        # sst2 = sst2.reshape(sst2.shape[2],sst2.shape[0],sst2.shape[1])
        # sst1 = tfs.ToTensor()(sst1)
        # sst2=tfs.ToTensor()(sst2)
        # gt=tfs.ToTensor()(gt)
        gt_ = np.zeros_like(gt)
        gt_[gt==255] = 1
        # aa= torch.from_numpy(np.array((gt_).max()))
        
        # gt_ind = tfs.ToTensor()((gt_).max)
        label = torch.nn.functional.one_hot(torch.from_numpy(np.array((gt_).max())).to(torch.int64), num_classes=2)
        if np.unique(gt_)[0]==1:
            # print('Got it!')
            label[0] = 0
        else:
            label[0] = 1
        # print(label)
        # gt_ind = torch.squeeze(torch.from_numpy(gt_),0)
        # print(gt_ind)
        # valid_mask = torch.nn.functional.one_hot(torch.from_numpy(gt_).to(torch.int64), num_classes=2)
        
        # print(torch.max(valid_mask[:,:,0]))
        # print(torch.min(valid_mask[:,:,1]))
        valid_mask = torch.ones((label.shape[0]+1, img.shape[1], img.shape[2]))
        # valid_mask = valid_mask * label.unsqueeze(-1).unsqueeze(-1)
        
        # label = label[1:]


        # front = gt2[:,:,0]
        # back = gt2[:,:,1]
        # # print(torch.max(gt2[:,:,0]))
        # # print(torch.max(gt2[:,:,1]))
        # gt2 = gt2.to(torch.float)
        # gt2 = gt2.reshape(gt2.shape[2],gt2.shape[0],gt2.shape[1])
 
        
        # gt_ind = gt.max().cpu().detach().numpy()
        # # print(gt_ind)
        # gt_ind = torch.LongTensor(gt_ind)
        # label = torch.nn.functional.one_hot(gt_ind, num_classes=2)


        return {'name': name, 'img': img, 'valid_mask': valid_mask, 'label': label}

    def __len__(self):
        return len(self.sst2_images)


class CDDataset_ir(data.Dataset):
    def __init__(self,path_root="./dataset/datasetLandslide/",mode="train"):
        super(CDDataset_ir,self).__init__()
        self.path_root=path_root+mode
        self.sst1_images_dir=os.listdir(os.path.join(self.path_root,"sst1"))
        self.sst1_images=[os.path.join(self.path_root,"sst1",img) for img in self.sst1_images_dir]
        self.sst2_images_dir = os.listdir(os.path.join(self.path_root, "sst2"))
        self.sst2_images = [os.path.join(self.path_root, "sst2", img) for img in self.sst2_images_dir]
        self.gt_images_dir=os.listdir(os.path.join(self.path_root,"labelnew"))
        self.gt_images=[os.path.join(self.path_root,"labelnew",img) for img in self.gt_images_dir]
        self.img_normal = TorchvisionNormalize()

    def __getitem__(self, item):
        name = os.path.split(self.sst2_images[item])[-1]
        name = os.path.splitext(name)[0]
        
        img = np.asarray(Image.open(self.sst2_images[item]))
        gt = np.asarray(Image.open(self.gt_images[item]))

        # img=self.img_normal(img)
        
        # img = np.ascontiguousarray(imutils.HWC_to_CHW(img))
        
        gt_ = np.zeros_like(gt)
        gt_[gt==255] = 1

        label = torch.nn.functional.one_hot(torch.from_numpy(np.array((gt_).max())).to(torch.int64), num_classes=2)
        label[0] = 1

        valid_mask = torch.ones((label.shape[0]+1, img.shape[1], img.shape[2]))

        return {'name': name, 'img': img, 'valid_mask': valid_mask, 'label': label}

    def __len__(self):
        return len(self.sst2_images)

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.png')


    
class CDDatasetMSF(CDDataset):

    def __init__(self, path_root, mode,scales=(1.0, 0.5, 1.5, 2.0)):
        super().__init__(path_root,mode)
        self.scales = scales
        self.img_normal = TorchvisionNormalize()
    def __getitem__(self, item):
        name = os.path.split(self.sst2_images[item])[-1]
        name = os.path.splitext(name)[0]
        
        img = imageio.imread(self.sst2_images[item])

        gt = np.array(Image.open(self.gt_images[item]))
        gt_ = np.zeros_like(gt)
        gt_[gt==255] = 1
        label = torch.nn.functional.one_hot(torch.from_numpy(np.array((gt_).max())).to(torch.int64), num_classes=2)
        label[0]=1
        # label = label[1:]
        # label = label.unsqueeze(0)
        # print(label)
        
        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))

        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]
        
        # mask = PIL.Image.open(get_mask_path(name, self.voc12_root))

        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": label, "img_path": self.sst2_images[item]}


        return out


class CDSegmentationDataset(data.Dataset):

    def __init__(self, label_dir,crop_size, voc12_root,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_method = 'random'):

        # self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.sst2_images_dir = os.listdir(os.path.join(self.voc12_root, "sst2"))
        self.img_name_list = [os.path.splitext(img)[0] for img in self.sst2_images_dir]


        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = imageio.imread(get_img_path(name, self.voc12_root))
        label = imageio.imread(os.path.join(self.label_dir, name + '.png'))

        img = np.asarray(img)

        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        
        if self.crop_method == "random":
            (img, label), _ = imutils.random_crop((img, label), self.crop_size, (0, 255))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        label = label.astype(np.uint8)
        img = imutils.HWC_to_CHW(img)
        return {'name': name, 'img': img, 'label': label}

class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)

class CDAffinityDataset(CDSegmentationDataset):
    def __init__(self,  label_dir, crop_size, voc12_root,
                 indices_from, indices_to,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False, crop_method=None):
        super().__init__( label_dir, crop_size, voc12_root, rescale, img_normal, hor_flip, crop_method=crop_method)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        reduced_label = imutils.pil_rescale(out['label'], 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out
