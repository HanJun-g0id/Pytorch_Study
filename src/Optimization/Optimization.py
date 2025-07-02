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
        
model = NeuralNetwork()

learning_rate = 1e-3  # 학습률: 파라미터를 얼마나 크게 조정할지
batch_size = 64       # 배치 크기: 한 번에 학습에 사용할 데이터 수
epochs = 10           # 에폭 수: 전체 데이터셋을 몇 번 반복할지

loss_fn = nn.CrossEntropyLoss()  # 다중 분류용 손실 함수
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # SGD 옵티마이저

def train_loop(dataloader, model, loss_fn, optimizer):
# 이 함수는 한 에폭 동안 진행되는 ‘빡공 타임’이야.
    size = len(dataloader.dataset)
    model.train()  # 학습 모드 (Dropout, BatchNorm 등 활성화)
    for batch, (X, y) in enumerate(dataloader):
        # 1. 예측 및 손실 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 2. 역전파 (gradient 계산)
        loss.backward()
        # 3. 파라미터 업데이트
        optimizer.step()
        # 4. 변화도 초기화 (누적 방지)
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()  # 평가 모드 (Dropout, BatchNorm 등 비활성화)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():  # 추론 시에는 변화도 계산 X (속도↑, 메모리↓)
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
