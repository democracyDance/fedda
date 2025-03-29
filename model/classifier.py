import torch.nn as nn

#全连接分类头
#这里面选7是因为pacs数据集的每个域都有7个class：Dog, Elephant, Giraffe, Guitar, Horse, House, Person
class Classifier(nn.Module):
    def __init__(self, num_classes=7):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(x)
