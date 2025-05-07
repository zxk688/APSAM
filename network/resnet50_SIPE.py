import torch
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50
from resnetmodules import Upsample,ResidualConv

class Net(nn.Module):

    def __init__(self, num_cls=21):
        super(Net, self).__init__()

        self.num_cls = num_cls

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))

        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        
        # self.upsample_1 = Upsample(2048, 2048, 2, 2)
        # self.up_residual_conv1 = ResidualConv(2048, 1024, 1, 1)

        # self.upsample_2 = Upsample(1024, 1024, 2, 2)
        # self.up_residual_conv2 = ResidualConv(1024, 512, 1, 1)

        # self.upsample_3 = Upsample(512, 512, 2, 2)
        # self.up_residual_conv3 = ResidualConv(512, 256, 1, 1)

        # self.upsample_4 = Upsample(256, 256, 2, 2)
        # self.up_residual_conv4 = ResidualConv(256, 128, 1, 1)
        
        # self.side1 = nn.Conv2d(128, 128, 1, bias=False)
        # self.side2 = nn.Conv2d(256, 128, 1, bias=False)
        # self.side3 = nn.Conv2d(512, 256, 1, bias=False)
        # self.side4 = nn.Conv2d(1024, 256, 1, bias=False)
        
        # self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        # self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        # self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        # self.side4 = nn.Conv2d(2048, 256, 1, bias=False)

        
        self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 256, 1, bias=False)
        self.classifier = nn.Conv2d(2048, self.num_cls-1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.side1, self.side2, self.side3, self.side4])

    def get_seed(self, norm_cam, label, feature):#全是用16x16计算的
        
        n,c,h,w = norm_cam.shape

        # iou evalution
        seeds = torch.zeros((n,h,w,c)).cuda()
        feature_s = feature.view(n,-1,h*w)
        feature_s = feature_s/(torch.norm(feature_s,dim=1,keepdim=True)+1e-5)
        correlation = F.relu(torch.matmul(feature_s.transpose(2,1), feature_s),inplace=True).unsqueeze(1) #[n,1,h*w,h*w]
        # correlation = correlation/torch.max(correlation, dim=-1)[0].unsqueeze(-1) #[n,1,h*w,h*w]
        cam_flatten = norm_cam.view(n,-1,h*w).unsqueeze(2) #[n,21,1,h*w]
        inter = (correlation * cam_flatten).sum(-1)
        union = correlation.sum(-1) + cam_flatten.sum(-1) - inter
        miou = (inter/union).view(n,self.num_cls,h,w) #[n,21,h,w]
        miou[:,0] = miou[:,0]*0.5
        probs = F.softmax(miou, dim=1)
        belonging = miou.argmax(1)#16x16
        seeds = seeds.scatter_(-1, belonging.view(n,h,w,1), 1).permute(0,3,1,2).contiguous()
        
        seeds = seeds * label
        return seeds, probs
    
    def get_prototype(self, seeds, feature):
        n,c,h,w = feature.shape
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(1)  # seed:[n,21,1,h,w], feature:[n,1,c,h,w], crop_feature:[n,21,c,h,w]
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1,c,h,w), (1,1)).view(n, self.num_cls, c, 1, 1) # prototypes:[n,21,c,1,1]
        return prototype

    def reactivate(self, prototype, feature):
        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype, dim=2)) # feature:[n,1,c,h,w], prototypes:[n,21,c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam

    def forward(self, x, valid_mask):

        N, C, H, W = x.size()

        # forward
        x0 = self.stage0(x)
        x1 = self.stage1(x0)    # b,256,64,64
        x2 = self.stage2(x1)    # b,*2,/2,/2
        x3 = self.stage3(x2)    # b,*2,/2,/2
        x4 = self.stage4(x3)    # b,*2,/2,/2

        # conv_x4 = (self.up_residual_conv1(self.upsample_1(x4)))
        # conv_x3 = (self.up_residual_conv2(self.upsample_2(conv_x4)))
        # conv_x2 = (self.up_residual_conv3(self.upsample_3(conv_x3)))
        # conv_x1 = (self.up_residual_conv4(self.upsample_4(conv_x2)))


        side1 = self.side1(x1)#[128,64,64]
        side2 = self.side2(x2)#[128,32,32]
        side3 = self.side3(x3)#[256,16,16]
        side4 = self.side4(x4)#[256,16,16]
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape, 
        #       side1.shape, side2.shape, side3.shape, side4.shape)
        # torch.Size([32, 64, 128, 128]) torch.Size([32, 256, 128, 128]) torch.Size([32, 512, 64, 64]) torch.Size([32, 1024, 32, 32]) torch.Size([32, 2048, 32, 32]) 
        # torch.Size([32, 128, 128, 128]) torch.Size([32, 128, 64, 64]) torch.Size([32, 256, 32, 32]) torch.Size([32, 256, 32, 32])

        # side1 = self.side1(conv_x1.detach())#[128,64,64]
        # side2 = self.side2(conv_x2.detach())#[128,32,32]
        # side3 = self.side3(conv_x3.detach())#[256,16,16]
        # side4 = self.side4(conv_x4.detach())#[256,16,16]
        
        hie_fea = torch.cat([ F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        # sem_feature = conv_x1
        # cam = self.classifier(conv_x1)#16*16分辨率太低
        
        sem_feature = x4
        cam = self.classifier(x4)#16*16分辨率太低,这里进行分类

        score = F.adaptive_avg_pool2d(cam, 1)   # 这一操作用于生成最终的类别得分 score，该分数通常代表网络的分类输出

        # initialize background map
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)*valid_mask

        seeds, probs = self.get_seed(norm_cam.clone(), valid_mask.clone(), sem_feature.clone())#用的16x16的特征
        prototypes = self.get_prototype(seeds, hie_fea)
        IS_cam = self.reactivate(prototypes, hie_fea)#16x16 重激活的 CAM 可以帮助模型更精确地定位物体的区域，并且通过原型来指导最终的分类结果
        

        return {"score": score, "cam": norm_cam, "seeds": seeds, "prototypes": prototypes, "IS_cam": IS_cam, "probs":probs}
        '''
        返回多个结果：这个 return 语句返回了多个结果，包括：
        score: 类别的得分，通过自适应平均池化计算得出。
        cam: 经过处理和归一化的类激活图(CAM)。
        seeds:从 CAM 中提取的种子特征。
        prototypes: 每个类别的原型特征。
        IS_cam: 重激活的类激活图，经过原型引导的 CAM。
        probs: 种子特征的类别概率。
        '''

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    # 调用该方法可以返回模型不同部分的参数，后面可针对不同的层选用不同的方法优化
    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, num_cls):
        super(CAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

    def forward(self, x, label):

        x0 = self.stage0(x)
        x1 = self.stage1(x0)    # b,256,64,64
        x2 = self.stage2(x1)    # b,*2,/2,/2
        x3 = self.stage3(x2)    # b,*2,/2,/2
        x4 = self.stage4(x3)    # b,*2,/2,/2

        side1 = self.side1(x1)#[n,128,64,64]
        side2 = self.side2(x2)#[n,128,32,32]
        side3 = self.side3(x3)#[n,256,16,16]
        side4 = self.side4(x4)#[n,256,16,16]

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        cam = self.classifier(x4)
        cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        hie_fea = (hie_fea[0] + hie_fea[1].flip(-1)).unsqueeze(0)
        
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)

        seeds, _ = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), hie_fea.clone())
        prototypes = self.get_prototype(seeds, hie_fea)
        IS_cam = self.reactivate(prototypes, hie_fea)


        return norm_cam[0], IS_cam[0]
