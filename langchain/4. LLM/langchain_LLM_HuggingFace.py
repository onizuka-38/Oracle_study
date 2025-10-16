from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Hugging Face Hub에 로그인하기 위해 토큰을 로드합니다.
load_dotenv()

# 1. HuggingFacePipeline.from_model_id()를 사용하여 모델과 파이프라인을 한 번에 로드
# model_id: 사용할 모델의 ID
# task: 수행할 작업 (텍스트 생성)
# device_map="auto": 사용 가능한 장치(GPU/CPU)에 자동으로 모델 할당
# pipeline_kwargs: 파이프라인에 전달할 추가 인자 (예: 생성할 최대 토큰 수)
llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-3-4b-it",
    task="text-generation",
    device_map="auto",
    pipeline_kwargs={"max_new_tokens": 512},
)

# 2. 프롬프트 템플릿 및 체인 구성
prompt = PromptTemplate.from_template("질문: {question}\n\n답변:")
chain = prompt | llm

# 3. 체인 실행
question = "세상에서 가장 높은 산은 무엇인가요?"
result = chain.invoke({"question": question})

print("--- google/gemma-3-4b-it 로컬 모델 응답 ---")
# 모델이 생성한 전체 텍스트에서 프롬프트 부분을 제외하고 출력
# (from_model_id 방식은 프롬프트를 포함하여 출력하는 경우가 많음)
print(result.split("답변:")[1].strip())