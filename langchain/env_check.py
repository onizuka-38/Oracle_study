# check_env.py
from dotenv import load_dotenv
import os
import psycopg2
from huggingface_hub import InferenceClient
from langchain_openai import OpenAIEmbeddings

print("🔍 환경 변수 확인 중...\n")

# 1️⃣ .env 로드
load_dotenv()

# 2️⃣ 환경 변수 읽기
openai_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(f"✅ OPENAI_API_KEY: {'OK' if openai_key else '❌ 없음'}")
print(f"✅ HUGGINGFACEHUB_API_TOKEN: {'OK' if hf_token else '❌ 없음'}")

# 3️⃣ Hugging Face 연결 테스트
if hf_token:
    try:
        client = InferenceClient(token=hf_token)
        result = client.text_generation(
            model="google/flan-t5-base",
            prompt="한 줄로 요약해줘: 인공지능이란 무엇인가?",
            max_new_tokens=30
        )
        print("🤖 Hugging Face 응답:", result.strip())
    except Exception as e:
        print("❌ Hugging Face 오류:", e)

# 4️⃣ OpenAI Embedding 테스트
if openai_key:
    try:
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
        vector = emb.embed_query("테스트 문장")
        print(f"🧠 OpenAI 임베딩 길이: {len(vector)}")
    except Exception as e:
        print("❌ OpenAI 임베딩 오류:", e)

# 5️⃣ PostgreSQL 연결 테스트
try:
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        database="testdb",
        user="test",
        password="5748"
    )
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print("🗄️ PostgreSQL 연결 성공:", cur.fetchone()[0])
    cur.execute("SELECT * FROM pg_extension WHERE extname='vector';")
    print("📦 pgvector 상태:", cur.fetchone())
    cur.close()
    conn.close()
except Exception as e:
    print("❌ PostgreSQL 연결 오류:", e)

print("\n✅ 환경 점검 완료.")
