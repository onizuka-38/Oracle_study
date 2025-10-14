############### predit_test.py
import torch
import torch.nn as nn
import sys

# GPU(CUDA) 사용 가능 여부를 확인하고 장치를 설정합니다
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 모델 아키텍처를 정의합니다. 학습할 때 사용했던 모델 구조와 동일해야 합니다
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력(input)은 4개, 출력(output)은 1개인 선형 회귀 모델입니다
        self.linear = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        return self.linear(x)

# 모델을 생성하고 GPU로 이동시킵니다
model = LinearRegressionModel().to(device)

# 학습된 모델의 상태(가중치와 편향)를 불러옵니다.
try:
    # 저장된 모델 파일을 불러옵니다
    state = torch.load('data/saved_model.pt', map_location=device)
    model.load_state_dict(state)
    print("모델 'saved_model.pt'가 성공적으로 불러와졌습니다.")
except FileNotFoundError:
    print("오류: 'saved_model.pt' 파일을 찾을 수 없습니다.")
    print("학습된 모델 파일이 현재 디렉터리에 있는지 확인해 주세요.")
    sys.exit()  # 파일이 없으면 프로그램을 종료합니다

# 모델을 평가 모드로 전환합니다. (Dropout 등 비활성화)
model.eval()

# 사용자로부터 4가지 값을 입력받습니다.
print("\n배추 가격을 예측하기 위해 다음 4가지 값을 입력해 주세요.")
try:
    avg_temp = float(input("평균 기온 (avg_temp): "))
    min_temp = float(input("최저 기온 (min_temp): "))
    max_temp = float(input("최고 기온 (max_temp): "))
    rain_fall = float(input("강수량 (rain_fall): "))
except ValueError:
    print("오류: 유효한 숫자를 입력해 주세요.")
    sys.exit()

# 입력받은 값을 PyTorch 텐서로 변환합니다
# 모델 입력의 형태([1, 4])에 맞게 변환하고, GPU로 이동시킵니다
input_data = torch.tensor(
    [[avg_temp, min_temp, max_temp, rain_fall]],
    dtype=torch.float32,
    device=device
)

# 학습된 모델로 예측을 수행합니다.
# torch.no_grad()를 사용하여 기울기 계산을 비활성화합니다
# 예측 단계에서는 기울기 계산이 필요 없기 때문에 메모리를 절약하고 속도를 높입니다
with torch.no_grad():
    prediction = model(input_data)

# 예측 결과를 출력합니다
# .item()을 사용하여 텐서에서 스칼라 값만 가져와 출력합니다
print(f"\n입력 값에 대한 예측 배추 가격은 : {prediction.item():.2f} 입니다")
