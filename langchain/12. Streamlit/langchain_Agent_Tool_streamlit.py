import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from datetime import datetime
import pytz
import os

from langchain_community.utilities import GoogleSerperAPIWrapper
from youtube_search import YoutubeSearch
from langchain_community.document_loaders import YoutubeLoader
from typing import List

# 환경 변수 로드
load_dotenv()

# API 키 설정
# os.environ['SERPER_API_KEY'] = '63148eb34e8eaa515273622fd72c0987e1955a5b'

# ==================== 도구 정의 ====================

@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        print(result)
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"

@tool
def get_web_search(query: str) -> str:
    """
    웹 검색을 수행하는 함수 (Google Serper API 사용).

    Args:
        query (str): 검색어

    Returns:
        str: 검색 결과
    """
    print('-------- WEB SEARCH --------')
    print(f"Query: {query}")

    search = GoogleSerperAPIWrapper()
    result = search.run(query)
    
    return result

@tool
def get_news_search(query: str) -> str:
    """
    뉴스 검색을 수행하는 함수 (Google Serper API 사용).

    Args:
        query (str): 검색어

    Returns:
        str: 뉴스 검색 결과
    """
    print('-------- NEWS SEARCH --------')
    print(f"Query: {query}")

    search = GoogleSerperAPIWrapper(
        type="news",  # 뉴스 타입 지정
        k=5,          # 결과 개수
        gl="kr",      # 한국 지역
        hl="ko"       # 한국어
    )
    result = search.run(query)
    
    return result

@tool
def get_youtube_search(query: str, max_results: int = 5) -> str:
    """
    유튜브 검색을 한 뒤, 영상들의 정보와 내용을 반환하는 함수.

    Args:
        query (str): 검색어
        max_results (int): 최대 결과 수

    Returns:
        str: 검색 결과 (제목, URL, 자막 내용 포함)
    """
    print('-------- YOUTUBE SEARCH --------')
    print(f"Query: {query}")

    # YoutubeSearch로 영상 검색
    videos = YoutubeSearch(query, max_results=max_results).to_dict()

    # 1시간 이상의 영상은 스킵 (duration 길이로 필터링)
    videos = [video for video in videos if len(video.get('duration', '')) <= 5]

    results = []
    video_data = []  # 영상 정보를 저장할 리스트 추가
    
    for idx, video in enumerate(videos, 1):
        video_url = 'https://youtube.com' + video['url_suffix']
        
        try:
            # YoutubeLoader로 자막 가져오기
            loader = YoutubeLoader.from_youtube_url(
                video_url, 
                language=['ko', 'en']  # 한국어, 영어 자막 우선순위
            )
            
            docs = loader.load()
            content = docs[0].page_content if docs else "자막 없음"
            
            # 세션 상태에 저장할 영상 정보
            video_data.append({
                'title': video['title'],
                'url': video_url,
                'channel': video['channel'],
                'duration': video['duration'],
                'views': video['views']
            })
            
            # 결과 포맷팅
            result = f"""
            [{idx}] {video['title']}
            - URL: {video_url}
            - 채널: {video['channel']}
            - 길이: {video['duration']}
            - 조회수: {video['views']}
            - 자막 내용: {content[:500]}...
            """
            results.append(result)
        except Exception as e:
            print(f"Error loading video {video_url}: {e}")
            # 자막 로드 실패 시 해당 영상 건너뛰기
            continue
    
    # 영상 데이터를 세션 상태에 저장
    if video_data:
        st.session_state.youtube_videos = video_data
    
    return "\n".join(results) if results else "검색 결과가 없습니다."


# ==================== 에이전트 시스템 프롬프트 ====================

WEB_AGENT_PROMPT = """
당신은 웹 검색 전문 에이전트입니다.
사용자의 질문에 대해 웹에서 정보를 검색하고 정확한 답변을 제공합니다.
get_web_search 도구를 사용하여 검색하고, 검색 결과를 분석하여 핵심 정보를 요약해서 전달하세요.
"""

YOUTUBE_AGENT_PROMPT = """
당신은 유튜브 비디오 검색 전문 에이전트입니다.
사용자가 원하는 주제의 유튜브 영상을 찾아주고,
영상 제목, URL, 채널명, 자막 내용을 분석하여 제공합니다.
get_youtube_search 도구를 사용하여 검색하세요.
"""

NEWS_AGENT_PROMPT = """
당신은 뉴스 검색 전문 에이전트입니다.
최신 뉴스와 이슈를 검색하고 요약해서 전달합니다.
get_news_search 도구를 사용하여 최신 뉴스를 검색하세요.
"""

COORDINATOR_PROMPT = """
당신은 멀티 에이전트 시스템의 조정자입니다.
사용자를 돕기 위해 최선을 다하는 인공지능 봇입니다.

사용자의 질문을 분석하여 적절한 도구를 선택하세요:
- get_current_time: 현재 시각 조회
- get_web_search: 일반 웹 검색 (정보, 지식, 사실 확인)
- get_youtube_search: 유튜브 영상 검색 (튜토리얼, 리뷰, 강의, 영상)
- get_news_search: 뉴스 검색 (최신 뉴스, 속보, 이슈)

키워드 분석:
- '유튜브', '영상', '비디오', '동영상', '튜토리얼' → get_youtube_search
- '뉴스', '속보', '최신', '이슈' → get_news_search
- '시간', '시각', '몇 시' → get_current_time
- 그 외 일반 검색 → get_web_search
"""


# ==================== 멀티 에이전트 시스템 ====================

class MultiAgentSystem:
    """멀티 에이전트 시스템 클래스"""
    
    def __init__(self):
        # LLM 초기화 (GPT-4o-mini 사용)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        
        # 사용 가능한 도구 리스트
        self.tools = [get_current_time, get_web_search, get_youtube_search, get_news_search]
        
        # 도구 이름으로 빠르게 접근하기 위한 딕셔너리
        self.tool_dict = {
            "get_current_time": get_current_time,
            "get_web_search": get_web_search,
            "get_youtube_search": get_youtube_search,
            "get_news_search": get_news_search
        }
        
        # LLM에 도구 바인딩 (LLM이 도구를 호출할 수 있도록 설정)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    def detect_agent_type(self, user_input: str) -> str:
        """
        사용자 입력을 분석하여 에이전트 타입 결정
        
        Args:
            user_input: 사용자가 입력한 텍스트
            
        Returns:
            에이전트 타입 문자열 ("YouTube", "News", "Time", "Web")
        """
        user_input_lower = user_input.lower()
        
        # 키워드 기반 라우팅
        if any(keyword in user_input_lower for keyword in ['유튜브', 'youtube', '영상', '비디오', '동영상', '튜토리얼']):
            return "YouTube"
        elif any(keyword in user_input_lower for keyword in ['뉴스', 'news', '속보', '최신', '이슈']):
            return "News"
        elif any(keyword in user_input_lower for keyword in ['시간', '시각', '몇 시']):
            return "Time"
        else:
            return "Web"
    
    def get_system_prompt(self, agent_type: str) -> str:
        """
        에이전트 타입에 따른 시스템 프롬프트 반환
        
        Args:
            agent_type: 에이전트 타입
            
        Returns:
            시스템 프롬프트 문자열
        """
        prompts = {
            "YouTube": YOUTUBE_AGENT_PROMPT,
            "News": NEWS_AGENT_PROMPT,
            "Web": WEB_AGENT_PROMPT,
            "Time": COORDINATOR_PROMPT
        }
        return prompts.get(agent_type, COORDINATOR_PROMPT)
    
    def process_response(self, messages):
        """
        AI 응답 처리 (도구 호출 포함)
        
        Args:
            messages: 대화 메시지 리스트
            
        Yields:
            AI 응답 청크 (스트리밍)
        """
        # LLM 응답을 스트리밍 방식으로 받기
        response = self.llm_with_tools.stream(messages)
        
        # 스트리밍된 청크들을 모으기 위한 변수
        gathered = None
        
        # 스트리밍 청크를 하나씩 처리
        for chunk in response:
            yield chunk  # Streamlit에 청크 전달 (실시간 출력)
            
            # 첫 청크이면 gathered에 저장
            if gathered is None:
                gathered = chunk
            else:
                # 이후 청크는 누적
                gathered += chunk
        
        # 도구 호출이 있는지 확인
        if gathered and gathered.tool_calls:
            # AI의 도구 호출 메시지를 대화 기록에 추가
            st.session_state.messages.append(gathered)
            
            # 각 도구 호출을 순차적으로 실행
            for tool_call in gathered.tool_calls:
                tool_name = tool_call['name']
                selected_tool = self.tool_dict[tool_name]
                
                # 도구 실행 및 결과 받기
                tool_msg = selected_tool.invoke(tool_call)
                
                # 도구 실행 결과를 대화 기록에 추가
                st.session_state.messages.append(tool_msg)
                
                # Streamlit UI에 도구 실행 완료 표시
                with st.chat_message("tool"):
                    st.caption(f"🔧 {tool_name} 실행 완료")
            
            # 도구 실행 결과를 포함하여 최종 응답 생성 (재귀 호출)
            for chunk in self.process_response(st.session_state.messages):
                yield chunk


# ==================== Streamlit UI ====================

# 페이지 설정 (반드시 첫 줄에 위치해야 함)
st.set_page_config(
    page_title="멀티 에이전트 챗봇",  # 브라우저 탭 제목
    page_icon="",                    # 브라우저 탭 아이콘
    layout="wide"                    # 와이드 레이아웃 사용
)

# 메인 타이틀 표시
st.title("멀티 에이전트 검색 챗봇")

# ==================== 사이드바 ====================
with st.sidebar:
    st.header("설정")
    
    # 에이전트 종류 안내
    st.markdown("""
    ### 사용 가능한 에이전트
    - **Web Agent**: 일반 정보 검색
    - **YouTube Agent**: 유튜브 영상 검색
    - **News Agent**: 최신 뉴스 검색
    - **Time Agent**: 현재 시각 조회
    """)
    
    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        # 메시지 초기 상태로 리셋
        st.session_state.messages = [
            SystemMessage(COORDINATOR_PROMPT),
            AIMessage("안녕하세요! 무엇을 도와드릴까요?")
        ]
        # 유튜브 비디오 세션 상태도 초기화
        if 'youtube_videos' in st.session_state:
            st.session_state.youtube_videos = []
        # 페이지 새로고침
        st.rerun()

# ==================== 세션 상태 초기화 ====================

# 멀티 에이전트 시스템 초기화 (한 번만 실행)
if "agent_system" not in st.session_state:
    st.session_state.agent_system = MultiAgentSystem()

# 메시지 기록 초기화 (한 번만 실행)
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(COORDINATOR_PROMPT),  # 시스템 프롬프트
        AIMessage("안녕하세요! 무엇을 도와드릴까요?")  # 초기 인사 메시지
    ]

# 유튜브 비디오 리스트 초기화
if "youtube_videos" not in st.session_state:
    st.session_state.youtube_videos = []

# ==================== 대화 기록 표시 ====================

# 저장된 모든 메시지를 순회하며 화면에 표시
for msg in st.session_state.messages:
    # 내용이 있는 메시지만 표시
    if msg.content:
        if isinstance(msg, SystemMessage):
            # 시스템 메시지는 사용자에게 보이지 않음
            continue
        elif isinstance(msg, AIMessage):
            # AI 메시지는 assistant 아이콘으로 표시
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            # 사용자 메시지는 user 아이콘으로 표시
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, ToolMessage):
            # 도구 실행 결과는 펼칠 수 있는 expander로 표시
            with st.expander("도구 실행 결과"):
                st.text(msg.content)

# ==================== 사용자 입력 처리 ====================

# 채팅 입력창 (화면 하단에 고정)
if prompt := st.chat_input("메시지를 입력하세요..."):
    
    # 1. 사용자 입력을 분석하여 적절한 에이전트 타입 결정
    agent_type = st.session_state.agent_system.detect_agent_type(prompt)
    
    # 2. 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(prompt)
        # 어떤 에이전트가 활성화되었는지 표시
        st.caption(f"{agent_type} Agent 활성화")
    
    # 3. 사용자 메시지를 대화 기록에 추가
    st.session_state.messages.append(HumanMessage(prompt))
    
    # 4. 에이전트 타입에 맞는 시스템 프롬프트로 업데이트
    #    (첫 번째 메시지인 SystemMessage를 동적으로 변경)
    system_prompt = st.session_state.agent_system.get_system_prompt(agent_type)
    st.session_state.messages[0] = SystemMessage(system_prompt)
    
    # 5. AI 응답 생성 (스트리밍 방식)
    response = st.session_state.agent_system.process_response(st.session_state.messages)
    
    # 6. AI 응답을 화면에 스트리밍으로 표시
    with st.chat_message("assistant"):
        # write_stream()은 제너레이터를 받아 실시간으로 텍스트 출력
        result = st.write_stream(response)
    
    # 7. 유튜브 영상이 있으면 임베드 표시
    if 'youtube_videos' in st.session_state and st.session_state.youtube_videos:
        with st.chat_message("assistant"):
            st.markdown("### 검색된 영상")
            for video in st.session_state.youtube_videos:
                with st.expander(f"{video['title']}", expanded=True): 
                    # 유튜브 영상 임베드
                    st.video(video['url'])
                    st.caption(f"채널: {video['channel']}")
                    st.caption(f"길이: {video['duration']}")
                    st.caption(f"조회수: {video['views']}")
                    st.caption(f"[YouTube에서 보기]({video['url']})")
        
        # 영상 표시 후 세션 상태 초기화 (다음 검색을 위해)
        st.session_state.youtube_videos = []
    
    # 8. AI의 최종 응답을 대화 기록에 추가
    st.session_state.messages.append(AIMessage(result))