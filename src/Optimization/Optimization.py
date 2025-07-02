import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",          # 데이터를 저장할 폴더
    train=True,           # 학습용 데이터셋
    download=True,        # 없으면 자동 다운로드
    transform=ToTensor()  # 이미지를 텐서로 변환
)

test_data = datasets.FashionMNIST(
    root="data",          # 저장 위치 동일
    train=False,          # 테스트용 데이터셋
    download=True,        # 없으면 자동 다운로드
    transform=ToTensor()  # 이미지를 텐서로 변환
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 2D 이미지를 1D 벡터로 펼침 (28x28 → 784)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),   # 입력: 784, 출력: 512
            nn.ReLU(),               # 비선형 활성화 함수
            nn.Linear(512, 512),     # 은닉층: 512 → 512
            nn.ReLU(),
            nn.Linear(512, 10),      # 출력: 512 → 10 (클래스 수)
        )

    def forward(self, x):
        x = self.flatten(x)                  # 이미지를 1차원 벡터로 변환
        logits = self.linear_relu_stack(x)   # 계층을 순서대로 통과
        return logits                        # 원시 예측값 반환
