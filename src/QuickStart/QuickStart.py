import torch  # 파이토치 기본 모듈
from torch import nn  # 신경망 모듈 (여기선 사용 안 함)
from torch.utils.data import DataLoader  # 데이터로더(배치 처리 등) 모듈
from torchvision import datasets  # 공개 데이터셋을 쉽게 쓸 수 있게 해주는 모듈
from torchvision.transforms import ToTensor  # 이미지를 텐서로 변환하는 함수

# FashionMNIST 학습 데이터셋을 다운로드하고, 이미지를 텐서로 변환해서 저장
training_data = datasets.FashionMNIST(
    root="data",           # 데이터를 저장할 폴더
    train=True,            # 학습용 데이터셋
    download=True,         # 없으면 자동 다운로드
    transform=ToTensor(),  # 이미지를 텐서로 변환
)

# FashionMNIST 테스트 데이터셋을 다운로드하고, 이미지를 텐서로 변환해서 저장
test_data = datasets.FashionMNIST(
    root="data",           # 데이터를 저장할 폴더
    train=False,           # 테스트용 데이터셋
    download=True,         # 없으면 자동 다운로드
    transform=ToTensor(),  # 이미지를 텐서로 변환
)

batch_size = 64  # 한 번에 불러올 데이터(배치) 크기

# 학습 데이터셋을 배치 단위로 불러오는 데이터로더 생성
train_dataloader = DataLoader(training_data, batch_size=batch_size)
# 테스트 데이터셋을 배치 단위로 불러오는 데이터로더 생성
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 테스트 데이터로더에서 첫 번째 배치만 꺼내서 X(이미지), y(정답 레이블) 확인
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # X: [배치크기, 채널수, 높이, 너비]
    print(f"Shape of y: {y.shape} {y.dtype}")      # y: [배치크기], 데이터 타입
    break  # 첫 배치만 확인하고 종료
