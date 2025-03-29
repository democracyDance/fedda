import torch.nn as nn
import torchvision.models as models
# 特征提取器，简单的cnn
class FeatureExtractor(nn.Module):
    '''def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(4096, 256)
        )

    def forward(self, x):
        return self.net(x)'''
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 去掉FC层
        self.fc = nn.Linear(2048, 256)  # 压缩为256维

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
