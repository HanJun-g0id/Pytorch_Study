# 사용할 연산 장치(GPU, MPS, CPU) 자동 선택
device = (
    "cuda"  # 만약 CUDA(GPU)가 사용 가능하면
    if torch.cuda.is_available()
    else "mps"  # 아니면 MPS(Apple Silicon GPU)가 사용 가능하면
    if torch.backends.mps.is_available()
    else "cpu"  # 둘 다 아니면 CPU 사용
)
print(f"Using {device} device")  # 어떤 장치 쓰는지 출력

# 신경망 모델 클래스를 정의 (nn.Module 상속)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # 부모 클래스 초기화
        self.flatten = nn.Flatten()  # 2D 이미지를 1D 벡터로 펼침
        self.linear_relu_stack = nn.Sequential(  # 여러 계층을 순서대로 쌓음
            nn.Linear(28*28, 512),  # 입력: 28*28=784, 출력: 512
            nn.ReLU(),              # 활성화 함수 ReLU
            nn.Linear(512, 512),    # 은닉층: 512 -> 512
            nn.ReLU(),              # 활성화 함수 ReLU
            nn.Linear(512, 10)      # 출력: 512 -> 10 (클래스 개수)
        )

    def forward(self, x):
        x = self.flatten(x)  # 이미지를 1차원으로 펼침
        logits = self.linear_relu_stack(x)  # 순차적으로 계층 통과
        return logits  # 최종 출력(로짓) 반환

model = NeuralNetwork().to(device)  # 모델 객체 생성 후 선택한 장치로 이동
print(model)  # 모델 구조 출력
