import streamlit as st
import numpy as np
import pandas as pd

# 1. 앱 제목 설정
st.title("인터랙티브 라인 차트")

# 2. 사이드바에 슬라이더 추가
st.sidebar.header("차트 옵션")
num_points = st.sidebar.slider(
    "데이터 포인트 개수를 선택하세요:",
    min_value=10,
    max_value=100,
    value=50  # 기본값
)

# 3. 슬라이더 값에 따라 데이터 생성
chart_data = pd.DataFrame(
    np.random.randn(num_points, 2),
    columns=['a', 'b']
)

# 4. 라인 차트 그리기
st.line_chart(chart_data)
st.write(f"{num_points}개의 포인트로 구성된 차트입니다.")