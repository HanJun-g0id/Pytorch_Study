import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 1. FashionMNIST 학습 데이터셋 다운로드 및 텐서 변환
training_data = datasets.FashionMNIST(
    root="data",         # 데이터 저장 경로
    train=True,          # 학습용 데이터
    download=True,       # 없으면 다운로드
    transform=ToTensor() # 이미지를 텐서로 변환
)

# 2. FashionMNIST 테스트 데이터셋 다운로드 및 텐서 변환
test_data = datasets.FashionMNIST(
    root="data",         # 데이터 저장 경로
    train=False,         # 테스트용 데이터
    download=True,       # 없으면 다운로드
    transform=ToTensor() # 이미지를 텐서로 변환
)

batch_size = 64  # 한 배치에 64개 데이터

# 3. DataLoader 생성 (배치 단위로 데이터 불러오기)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 4. DataLoader에서 한 배치 꺼내기
train_features, train_labels = next(iter(train_dataloader))

# 5. 배치 데이터 크기 출력
print(f"Feature batch shape: {train_features.size()}")  # 예: torch.Size([64, 1, 28, 28])
print(f"Labels batch shape: {train_labels.size()}")     # 예: torch.Size([64])

# 6. 첫 번째 이미지 시각화
img = train_features[0].squeeze()    # [1, 28, 28] → [28, 28]로 변환
label = train_labels[0].item()       # 라벨 값 추출
plt.imshow(img, cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
