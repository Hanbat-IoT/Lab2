from torch import nn
import torch.nn.functional as F

def get_model(dataset_name):
    if dataset_name == "cifar":
        return CNNCifar()
    elif dataset_name == "mnist":
        return CNNMNIST()

class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        # 라즈베리파이 CPU 속도를 고려하여 필터 개수는 적당히(32, 64) 조절했습니다.
        # 핵심: nn.BatchNorm2d()가 추가되어야 46% 구간을 돌파합니다.
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # 배치 정규화
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 32x32 이미지가 pool을 3번 거치면: 16 -> 8 -> 4 크기가 됨
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool 순서가 국룰입니다.
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 64 * 4 * 4) # 펼치기 (Flatten)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNNMNIST(nn.Module):
    def __init__(self):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Raw logits for CrossEntropyLoss
        # return F.log_softmax(x, dim=1)  # 제거!