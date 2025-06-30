import torch
from torch import nn

# 1. 학습/추론에 사용할 장치(GPU/MPS/CPU) 자동 선택
device = (
    "cuda" if torch.cuda.is_available()
    # 혹시 엔비디아 GPU(cuda)라는 초고속 작업대 쓸 수 있어?
    else "mps" if torch.backends.mps.is_available()
    # 그럼 애플 실리콘 GPU(mps) 작업대는 어때?
    else "cpu"
    # 둘 다 없으면 그냥 일반 책상(cpu)에서 하지 뭐
)
print(f"Using {device} device")

# 2. 신경망 모델 클래스 정의 (nn.Module 상속)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 이미지를 1차원 벡터로 펼침 (28x28 → 784)
        self.flatten = nn.Flatten()
        # 순차적으로 쌓인 선형 계층 + ReLU 활성화 함수
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # 입력: 784, 출력: 512
            # 선형 변환 블록. (이게 바로 행렬 곱셈(`@`)이 일어나는 곳!)
            nn.ReLU(),
            # 결과값 중에 음수는 그냥 0으로 만들어 버려!“라고 외치는 
            # ‘긍정왕’ 활성화 함수 블록. 
            # 모델에 비선형성을 추가해서 더 복잡한 문제를 풀 수 있게 해줘.
            
            nn.Linear(512, 512),    # 입력: 512, 출력: 512
            nn.ReLU(), # 이런 `Linear`와 `ReLU`를 여러 겹 쌓아서 모델의 깊이를 만드는 거야.
            nn.Linear(512, 10),     # 입력: 512, 출력: 10 (클래스 수)
        )

    def forward(self, x):
        x = self.flatten(x)               # 펼치기
        logits = self.linear_relu_stack(x) # 계층 통과
        return logits                     # 결과 반환

# 3. 모델 인스턴스 생성 및 장치로 이동
model = NeuralNetwork().to(device)
print(model)
