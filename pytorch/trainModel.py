############ trainModel.py #######################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# GPU(CUDA) 사용 가능 여부를 확인하고 장치를 설정합니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# price_data.csv 파일을 불러옵니다.
# (이 파일은 스크립트와 동일한 디렉토리에 있어야 합니다.)
try:
    data = pd.read_csv('data/price_data.csv', sep=',')
    xy = np.array(data, dtype=np.float32)
except FileNotFoundError:
    print("Error: 'price_data.csv' 파일을 찾을 수 없습니다. 파일을 스크립트와 같은 디렉토리에 넣어주세요.")
    exit()

# 데이터를 입력 변수(x)와 목표 변수(y)로 분리합니다.
# 4개의 변인을 입력으로 받습니다.
x_data = xy[:, 1:-1]
# 마지막 열의 가격을 목표 변수로 받습니다.
y_data = xy[:, [-1]]

# Numpy 배열을 PyTorch 텐서로 변환하고, GPU 장치로 이동시킵니다.
X_train = torch.from_numpy(x_data).to(device)
Y_train = torch.from_numpy(y_data).to(device)

# PyTorch 모델 정의
# 모델을 nn.Module 클래스로 정의합니다.
# nn.Module을 상속받는 클래스를 만들고,
# __init__ 메서드에서 모델의 레이어들을 정의하며,
# forward 메서드에서 데이터가 모델을 통과하는 방식을 정의합니다.
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력(input)은 4개, 출력(output)은 1개인 선형 회귀 모델입니다.
        self.linear = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        # x_data를 받아 선형 레이어를 통과시킵니다.
        return self.linear(x)

# 모델, 손실 함수, 옵티마이저를 정의합니다.
model = LinearRegressionModel().to(device)  # 모델을 GPU로 이동시킵니다.
cost_function = nn.MSELoss()                # 평균 제곱 오차(MSE)를 손실 함수로 사용합니다.
optimizer = optim.SGD(model.parameters(), lr=0.000005)  # 경사 하강법 옵티마이저를 사용합니다.

# 모델 학습 루프
for step in range(100001):
    # 1. 가설 계산 (순전파, Forward)
    hypothesis = model(X_train)

    # 2. 손실(cost) 계산
    cost = cost_function(hypothesis, Y_train)

    # 3. 기울기 초기화 (Gradient)
    # 이전 단계에서 계산된 기울기를 0으로 초기화합니다.
    optimizer.zero_grad()

    # 4. 역전파를 통해 기울기 계산 (Backward)
    # cost에 대한 모든 모델 변수(가중치와 편향)의 기울기를 자동으로 계산합니다.
    cost.backward()

    # 5. 가중치 업데이트 (Update)
    # 계산된 기울기를 사용하여 모델의 가중치와 편향을 업데이트합니다.
    optimizer.step()

    # 500 스텝마다 현재 상태를 출력합니다.
    if step % 500 == 0:
        # .item()을 사용하여 GPU 텐서에서 스칼라 값만 가져와 출력
        print(f"Step: {step:5}, Loss: {cost.item():10.2f}")

# 가설 텐서의 차원과 모양을 출력합니다.
print(f"\n가설 텐서의 차원 : {hypothesis.dim()}")
print(f"가설 텐서의 모양 : {hypothesis.shape}")

# 첫 번째 행의 예측 값을 출력합니다.
print(f"\n배추 가격 : {hypothesis[0].item():10.2f}, {hypothesis[2900].item():10.2f}")

# 학습된 모델을 저장합니다.
# PyTorch는 모델의 가중치만 저장하는 state_dict() 방식을 권장합니다.
torch.save(model.state_dict(), 'data/saved_model.pt')
print('\n학습된 모델을 저장했습니다. 파일명 : saved_model.pt')
