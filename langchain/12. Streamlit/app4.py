import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# --- 1. 기본 설정 ---

st.title("간단 AI 챗봇")

llm = ChatOpenAI(model="gpt-4o-mini")

# --- 2. 대화 기록을 위한 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. 이전 대화 내용 표시 ---
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)

# --- 4. 사용자 입력 처리 ---
if prompt := st.chat_input("무엇이든 물어보세요."):
    # 4-1. 사용자 메시지를 기록하고 화면에 표시
    st.session_state.messages.append(HumanMessage(content=prompt, role="user"))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4-2. 'AI'의 답변을 스트리밍으로 화면에 표시
    with st.chat_message("assistant"):
        # [핵심 변경] echo_stream 대신, 실제 LLM의 stream() 메서드를 호출합니다.
        # LLM에게 전체 대화 기록을 전달하여 답변을 생성하게 합니다.
        response = st.write_stream(llm.stream(st.session_state.messages))
    
    # 4-3. 스트리밍으로 완성된 전체 답변을 기록에 추가
    st.session_state.messages.append(AIMessage(content=response, role="assistant"))