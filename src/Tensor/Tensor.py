# 텐서 초기화 방법

#1. 데이터로부터 직접 생성
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)  # 리스트나 배열로부터 바로 생성

#2. 넘파이 배열로부터 생성
import numpy as np
np_array = np.array(data)
x_np = torch.from_numpy(np_array)  # 넘파이 배열을 텐서로 변환

#3. 다른 텐서로부터 생성
x_ones = torch.ones_like(x_data)  # x_data와 같은 크기, 값은 모두 1
x_rand = torch.rand_like(x_data, dtype=torch.float)  # x_data와 같은 크기, 랜덤 값

#4. 무작위/상수 값으로 생성
shape = (2, 3)
rand_tensor = torch.rand(shape)    # 0~1 사이 랜덤 값
ones_tensor = torch.ones(shape)    # 모두 1
zeros_tensor = torch.zeros(shape)  # 모두 0

# 텐서의 속성
tensor = torch.rand(3, 4)
print(tensor.shape)   # 텐서의 모양 (3행 4열)
print(tensor.dtype)   # 데이터 타입 (예: float32)
print(tensor.device)  # 저장된 장치 (CPU/GPU)

# 텐서 연산 예시

#1. 인덱싱 & 슬라이싱
tensor = torch.ones(4, 4)
print(tensor[0])        # 첫 번째 행
print(tensor[:, 0])     # 첫 번째 열
print(tensor[..., -1])  # 마지막 열
tensor[:, 1] = 0        # 두 번째 열을 0으로 변경
print(tensor)

#2. 텐서 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # 열 방향으로 합침
print(t1)

#3. 산술 연산
# 행렬 곱
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# 요소별 곱
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#4. 단일-요소 텐서
agg = tensor.sum() # 모든 원소 합
agg_item = agg.item() # 파이썬 숫자로 변환
print(agg_item, type(agg_item))

#5. 바꿔치기 연산
print(tensor)
tensor.add_(5) # 모든 원소에 5 더함 (원본 변경)
print(tensor)

# 텐서와 넘파이 변환
# 텐서를 넘파이로
t = torch.ones(5)
n = t.numpy()
t.add_(1)
print(t)  # tensor([2., 2., 2., 2., 2.])
print(n)  # [2. 2. 2. 2. 2.]
# 넘파이를 텐서로
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(t)  # tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
print(n)  # [2. 2. 2. 2. 2.]

# 텐서를 GPU로 옮기기
if torch.cuda.is_available():
    tensor = tensor.to("cuda")  # GPU로 이동
