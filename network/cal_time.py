import time
import torch
import resnet50
import torch.nn as nn

# 创建模型
model = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))
model.eval()  # 设置为推理模式

# 生成模拟数据：10张256x256的图片，假设batch size为10
images = torch.randn(10, 3, 256, 256)  # 10张RGB图像，尺寸为256x256

# 如果使用GPU，将数据和模型移到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
images = images.to(device)

# 记录推理时间
start_time = time.time()

# 执行推理
with torch.no_grad():  # 不计算梯度
    outputs = model(images)

# 计算并打印推理时间
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time for 10 images: {inference_time:.4f} seconds")
