from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# API 키 로드
load_dotenv()

# 모델 정의
model = ChatOpenAI(model="gpt-4o-mini")

# 체인 1: 리뷰 분석기
prompt_analyze = ChatPromptTemplate.from_template(
    """당신은 영화 리뷰 분석 전문가입니다.
    주어진 리뷰에서 영화 제목과 리뷰의 감정(긍정/부정)을 추출하여 JSON 형식으로 반환해주세요.
    
    반드시 다음 형식으로 반환하세요:
    {{"title": "영화제목", "sentiment": "긍정 또는 부정"}}

    리뷰: {review}
    """
)
parser_json = JsonOutputParser()

# 분석 체인 정의
chain_analyze = prompt_analyze | model | parser_json

# 체인 2: 답변 작성기
prompt_compose = ChatPromptTemplate.from_template(
    """당신은 친절한 리뷰 답변 챗봇입니다.
    분석된 영화 제목과 감정을 바탕으로 사용자에게 보낼 답변을 작성해주세요.
    - 감정이 '긍정'이면: 감사 인사를 표현하세요.
    - 감정이 '부정'이면: 유감을 표현하세요.

    영화 제목: {title}
    감정: {sentiment}
    """
)

# 답변 작성 체인 정의
chain_compose = prompt_compose | model | StrOutputParser()

# 전체 멀티 체인 정의
overall_chain = chain_analyze | chain_compose

# 실행할 리뷰
review_text = "영화 인셉션을 봤는데, 스토리가 너무 복잡하고 지루했어요."

# 전체 체인 실행
result = overall_chain.invoke({"review": review_text})

# 최종 결과 출력
print(result)