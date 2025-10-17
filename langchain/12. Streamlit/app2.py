
import streamlit as st
import pandas as pd
import numpy as np

# 1. 앱 제목 설정
st.title("데이터프레임 필터링")

# 2. 샘플 데이터 생성
@st.cache_data  # 데이터를 캐싱하여 재실행 속도 향상
def get_data():
    return pd.DataFrame({
        'Category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'Value': np.random.randint(1, 100, 6)
    })
df = get_data()

# 3. 사이드바에 selectbox 위젯 추가
st.sidebar.header("필터 옵션")
categories = ['All'] + list(df['Category'].unique())
selected_category = st.sidebar.selectbox("카테고리를 선택하세요:", categories)

# 4. 선택된 카테고리로 데이터 필터링
if selected_category == 'All':
    filtered_df = df
else:
    filtered_df = df[df['Category'] == selected_category]

# 5. 결과 표시
st.write(f"### '{selected_category}' 카테고리 데이터")
st.dataframe(filtered_df)