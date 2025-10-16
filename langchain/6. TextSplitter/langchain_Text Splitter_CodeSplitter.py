from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

python_code = """
def hello_world():
    print("Hello, World!")

class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, {self.name}")
"""
# RecursiveCharacterTextSplitter의 언어별 사전 설정(preset)을 사용
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=100, chunk_overlap=0
)

chunks = text_splitter.split_text(python_code)

print(f"--- CodeSplitter (Python) (총 {len(chunks)}개) ---")
for i, chunk in enumerate(chunks):
    print(f"[{i+1}] {chunk}")