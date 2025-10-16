from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- 1. 기본 설정 ---
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()


# --- 2. 퓨샷 프롬프트 구성 요소 정의 ---
# LLM에게 보여줄 모범 답안 목록 (examples)을 정의합니다.
examples = [
    {"input": "행복", "output": "불행"},
    {"input": "큰", "output": "작은"},
    {"input": "밝은", "output": "어두운"},
]

# 예시(examples)를 어떤 형식으로 보여줄지 (example_prompt) 정의합니다.
example_prompt = PromptTemplate(
    input_variables=["input", "output"], # 예시에서 사용할 변수 이름들
    template="입력: {input}\n출력: {output}",
)

# 퓨샷 프롬프트 템플릿을 최종적으로 조립합니다.
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="다음 단어의 반의어를 출력하세요:",
    suffix="입력: {user_input}\n출력:",
    # 최종적으로 사용자가 입력할 변수의 이름을 지정합니다.
    input_variables=["user_input"], 
)


# --- 3. 체인 생성 및 실행 ---
# LCEL을 사용하여 모든 구성 요소를 연결합니다.
chain = few_shot_prompt | model | parser

# 'input_variables'에 지정한 "user_input" 이름으로 실제 질문을 전달하여 체인을 실행합니다.
result = chain.invoke({"user_input": "뜨거운"})

# 결과를 출력합니다.
print(result)