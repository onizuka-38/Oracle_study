from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

class Movie(BaseModel):
    title: str = Field(description="영화 제목")
    director: str = Field(description="감독 이름")

# (이전 코드와 동일한 import 및 model 설정)
parser = PydanticOutputParser(pydantic_object=Movie)
prompt = ChatPromptTemplate.from_template(
    """영화 정보를 추출해줘.\n{format_instructions}\n질문: 영화 인셉션의 감독은 누구야?"""
)

# 전: 파서 미적용 (모델의 원본 출력)
chain_before = prompt | model | StrOutputParser()
result_before = chain_before.invoke({"format_instructions": parser.get_format_instructions()})

# 후: PydanticOutputParser 적용
chain_after = prompt | model | parser
result_after = chain_after.invoke({"format_instructions": parser.get_format_instructions()})


print("--- [전] 파서 미적용 (LLM 원본 출력) ---")
print(result_before)
print(f"타입: {type(result_before)}\n")

print("--- [후] PydanticOutputParser 적용 ---")
print(result_after)
print(f"타IP: {type(result_after)}")