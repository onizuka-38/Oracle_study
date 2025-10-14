import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# optimizer 설정. 경사 하강법 SGD를 사용하고 learing rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs):
    # 모델의 forward 함수를 사용하여 모델의 출력값을 구함
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train) # 평균 제곱 오차 함수(pytorch내장)
    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()
    
    if epoch % 100 == 0:
        W = model.linear.weight.item()
        b = model.linear.bias.item()

        # 100번마다 log 출력
        print('Epoch: {:4d}, W: {:.6f}, b: {:.6f}, cost: {:.6f}'.format(epoch, W, b, cost.item()))