import torch

x = torch.ones(5)     # 입력 텐서
y = torch.zeros(3)    # 기대 출력
w = torch.randn(5, 3, requires_grad=True)  # 가중치 (gradient 추적 활성화)
b = torch.randn(3, requires_grad=True)      # 편향 (gradient 추적 활성화)

z = torch.matmul(x, w) + b  # 순전파 연산
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)  # 손실 함수

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()  # 역전파 실행

print(w.grad)  # ∂loss/∂w 저장됨
print(b.grad)  # ∂loss/∂b 저장됨

# 추적 비활성화 방법 1: torch.no_grad()
with torch.no_grad():
    z = torch.matmul(x, w) + b  # 연산 기록 X (z.requires_grad=False)

# 추적 비활성화 방법 2: .detach()
z_det = z.detach()  # 기존 텐서의 복사본 생성 (추적 X)

loss.backward()  # 첫 실행: w.grad = 4
loss.backward()  # 두 번째 실행: w.grad = 8 (누적됨)

w.grad.zero_()   # 수동으로 초기화 필요

out.backward(torch.ones_like(out))  # vᵀ ⋅ J 계산
# `torch.ones_like(out)`은
# `out`과 똑같은 모양이면서 모든 값이 1인 텐서 제작 
# 이걸 ’질문 벡터 v’로 사용
