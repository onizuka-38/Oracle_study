from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- 1. 기본 설정 ---
load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# --- 2. 퓨샷 채팅 프롬프트 구성 요소 ---
# 모범 답안 (examples) 정의
examples = [
    {"input": "Hi!", "output": "안녕하십니까."},
    {"input": "How are you?", "output": "어떻게 지내십니까?"},
]

# 각 예시를 어떤 '대화' 형식으로 보여줄지 정의
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# 퓨샷 채팅 프롬프트 템플릿 조립
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 최종 프롬프트 = 시스템 메시지 + 예시들 + 사용자 입력
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 영어를 한국어의 정중한 '하십시오체'로 번역하는 AI입니다."),
        few_shot_prompt,  # 예시들이 여기에 삽입됨
        ("human", "{user_input}"),
    ]
)

# --- 3. 체인 생성 및 실행 ---
chain = final_prompt | model | parser

result = chain.invoke({"user_input": "See you later."})

print(result)