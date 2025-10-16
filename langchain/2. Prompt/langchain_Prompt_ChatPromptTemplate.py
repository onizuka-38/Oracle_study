from langchain.prompts import ChatPromptTemplate

# (역할, 템플릿) 튜플의 리스트로 템플릿을 정의합니다.
chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {language} 전문가입니다."),
    ("human", "{text} 문장의 문법을 교정해주세요.")
])

# 변수를 채워 메시지 리스트를 생성합니다.
messages = chat_template.format_messages(language="영어", text="He don't like apple.")

print(messages)
print(type(messages))