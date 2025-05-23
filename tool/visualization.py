import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def color_pro(pro, img=None, mode='hwc'):
	H, W = pro.shape
	pro_255 = (pro*255).astype(np.uint8)
	pro_255 = np.expand_dims(pro_255,axis=2)
	color = cv2.applyColorMap(pro_255,cv2.COLORMAP_JET)
	color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
	if img is not None:
		rate = 0.5
		if mode == 'hwc':
			assert img.shape[0] == H and img.shape[1] == W
			color = cv2.addWeighted(img,rate,color,1-rate,0)
		elif mode == 'chw':
			assert img.shape[1] == H and img.shape[2] == W
			img = np.transpose(img,(1,2,0))
			color = cv2.addWeighted(img,rate,color,1-rate,0)
			color = np.transpose(color,(2,0,1))
	else:
		if mode == 'chw':
			color = np.transpose(color,(2,0,1))	
	return color
		
def generate_vis(p, gt, img, func_label2color, threshold=0.1, norm=True):
	# All the input should be numpy.array 
	# img should be 0-255 uint8
	C, H, W = p.shape

	if norm:
		prob = max_norm(p, 'numpy')
	else:
		prob = p
	if gt is not None:
		prob = prob * gt
	prob[prob<=0] = 1e-7
	if threshold is not None:
		prob[0,:,:] = np.power(1-np.max(prob[1:,:,:],axis=0,keepdims=True), 4)

	CAM = ColorCAM(prob, img)
	
	return CAM

def max_norm(p, version='torch', e=1e-5):
	if version == 'torch':
		if p.dim() == 3:
			C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
			min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
		elif p.dim() == 4:
			N, C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
	elif version == 'numpy' or version == 'np':
		if p.ndim == 3:
			C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(1,2),keepdims=True)
			min_v = np.min(p,(1,2),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
		elif p.ndim == 4:
			N, C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(2,3),keepdims=True)
			min_v = np.min(p,(2,3),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
	return p

def ColorCAM(prob, img):
	assert prob.ndim == 3
	C, H, W = prob.shape
	colorlist = []
	for i in range(C):
		colorlist.append(color_pro(prob[i,:,:],img=img,mode='chw'))
	CAM = np.array(colorlist)/255.0
	return CAM
	
def ColorCLS(prob, func_label2color):
	assert prob.ndim == 3
	prob_idx = np.argmax(prob, axis=0)
	CLS = func_label2color(prob_idx).transpose((2,0,1))
	return CLS
	
def VOClabel2colormap(label):
	m = label.astype(np.uint8)
	r,c = m.shape
	cmap = np.zeros((r,c,3), dtype=np.uint8)
	cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
	cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
	cmap[:,:,2] = (m&4)<<5
	cmap[m==255] = [255,255,255]
	return cmap

def dense_crf(probs, img=None, n_classes=21, n_iters=1, scale_factor=1):
	c,h,w = probs.shape
	if img is not None:
		assert(img.shape[1:3] == (h, w))
		img = np.transpose(img,(1,2,0)).copy(order='C')

	d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.

	unary = unary_from_softmax(probs)
	unary = np.ascontiguousarray(unary)
	d.setUnaryEnergy(unary)
	d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
	d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
	Q = d.inference(n_iters)

	preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w))
	return preds


def convert_to_tf(img):
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	img_8 = img.detach().cpu().numpy().transpose((1,2,0))
	img_8 = np.ascontiguousarray(img_8)
	img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
	img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
	img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
	img_8[img_8 > 255] = 255
	img_8[img_8 < 0] = 0
	img_8 = img_8.astype(np.uint8)
	img_8 = cv2.resize(img_8, (img_8.shape[1]//2, img_8.shape[0]//2)).transpose((2,0,1))

	return img_8
