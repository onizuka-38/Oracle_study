
import streamlit as st

# 1. 앱 제목 설정
st.title("간단한 인사 앱 👋")

# 2. 사용자에게 텍스트 입력받기
name = st.text_input("당신의 이름은 무엇인가요?")

# 3. 버튼을 누르면 조건문이 True가 됨
if st.button("인사하기"):
    if name:
        st.write(f"안녕하세요, **{name}**님! 반갑습니다.")
    else:
        st.write("이름을 입력해주세요.")