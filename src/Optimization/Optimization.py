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
