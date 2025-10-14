import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 랜덤 시드 고정
torch.manual_seed(1)

# 학습 데이터 정의
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 이진 분류 모델 정의
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# 모델 생성
model = BinaryClassifier()

# optimizer 설정 (경사 하강법 SGD)
optimizer = optim.SGD(model.parameters(), lr=1)

# 학습
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # 1. H(x) 계산
    hypothesis = model(x_train)

    # 2. cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # 3. cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        # 예측값이 0.5 이상이면 True(1), 아니면 False(0)
        prediction = hypothesis >= torch.FloatTensor([0.5])
        # 실제값과 비교하여 정확도 계산
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)

        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'
              .format(epoch, nb_epochs, cost.item(), accuracy * 100))

# 학습된 모델에 임의의 입력 [1, 4]를 적용하여 검증
new_var = torch.FloatTensor([[1.0, 4.0]])
pred_y = model(new_var)  # forward 연산
print("\n훈련 후 입력이 [1, 4]일 때의 예측값:", pred_y)

# 예측 결과가 0.5 이상이면 True, 아니면 False
prediction = pred_y >= torch.FloatTensor([0.5])
print("분류 결과:", prediction)