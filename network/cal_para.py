import resnet50
import torch
import torchvision.models as models

model = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")