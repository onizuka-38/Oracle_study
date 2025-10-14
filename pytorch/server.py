########## server.py
from flask import Flask, request, render_template_string
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
        self.linear = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        return self.linear(x)

# 모델을 생성하고 GPU로 이동시킵니다
model = LinearRegressionModel().to(device)

# 학습된 모델의 상태(가중치와 편향)를 불러옵니다.
try:
    model.load_state_dict(torch.load('data/saved_model.pt', map_location=device))
    print("모델 'saved_model.pt'가 성공적으로 불러와졌습니다")
except FileNotFoundError:
    print("오류: 'saved_model.pt' 파일을 찾을 수 없습니다. 학습된 모델 파일이 현재 디렉터리에 있는지 확인해 주세요")
    sys.exit()

# 모델을 평가 모드로 전환합니다. (Dropout 등 비활성화)
model.eval()

# Flask 웹 애플리케이션을 초기화합니다
app = Flask(__name__)

# 메인 페이지를 정의합니다.
@app.route('/', methods=['GET', 'POST'])
def home():
    # POST 요청(폼 제출)인 경우
    if request.method == 'POST':
        try:
            # 폼 데이터를 가져와 float으로 변환합니다
            avg_temp = float(request.form['avg_temp'])
            min_temp = float(request.form['min_temp'])
            max_temp = float(request.form['max_temp'])
            rain_fall = float(request.form['rain_fall'])

            # 입력값을 PyTorch 텐서로 변환합니다
            input_data = torch.tensor(
                [[avg_temp, min_temp, max_temp, rain_fall]],
                dtype=torch.float32,
                device=device
            )

            # 예측을 수행합니다.
            with torch.no_grad():
                prediction = model(input_data)
            predicted_price = f"{prediction.item():.2f}"

            # 예측 결과를 포함한 HTML을 반환합니다
            return render_template_string(HTML_TEMPLATE, prediction=predicted_price)
        except (ValueError, KeyError):
            return '유효한 숫자를 입력해 주세요. <a href="/">다시 시도</a>'

    # GET 요청(페이지 접근)인 경우
    return render_template_string(HTML_TEMPLATE, prediction=None)

# 웹 페이지의 HTML 템플릿입니다
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>배추 가격 예측</title>
<style>
  body { font-family: sans-serif; display:flex; justify-content:center; align-items:center; height:100vh; background-color:#f0f4f8; }
  .container { background:#fff; padding:2rem; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1); text-align:center; width: min(420px, 92vw); }
  h1 { color:#333; margin-top:0; }
  .form-group { margin-bottom:1rem; text-align:left; }
  label { display:block; margin-bottom:0.5rem; color:#555; }
  input { width:100%; padding:0.5rem; border:1px solid #ccc; border-radius:5px; box-sizing:border-box; }
  button { margin-top:0.5rem; padding:0.75rem 1.5rem; background-color:#007bff; color:white; border:none; border-radius:5px; cursor:pointer; font-size:1rem; }
  button:hover { background-color:#0056b3; }
  .result { margin-top:1.5rem; padding:1rem; border:2px solid #007bff; border-radius:5px; }
</style>
</head>
<body>
  <div class="container">
    <h1>배추 가격 예측</h1>
    <form method="post">
      <div class="form-group">
        <label for="avg_temp">평균 기온</label>
        <input type="text" id="avg_temp" name="avg_temp" required>
      </div>
      <div class="form-group">
        <label for="min_temp">최저 기온</label>
        <input type="text" id="min_temp" name="min_temp" required>
      </div>
      <div class="form-group">
        <label for="max_temp">최고 기온</label>
        <input type="text" id="max_temp" name="max_temp" required>
      </div>
      <div class="form-group">
        <label for="rain_fall">강수량</label>
        <input type="text" id="rain_fall" name="rain_fall" required>
      </div>
      <button type="submit">예측하기</button>
    </form>

    {% if prediction %}
    <div class="result">
      <h3>예측 결과</h3>
      <p><strong>예상 배추 가격 :</strong> {{ prediction }} 원</p>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

if __name__ == '__main__':
    # 웹 서버를 5000번 포트로 실행합니다
    app.run(host='0.0.0.0', port=5000, debug=True)
