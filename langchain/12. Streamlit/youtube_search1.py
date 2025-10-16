# app.py
import streamlit as st
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi

st.title("🎬 유튜브 영상 요약기")

query = st.text_input("검색할 키워드를 입력하세요:")

if query:
    results = YoutubeSearch(query, max_results=3).to_dict()
    for video in results:
        title = video['title']
        url = f"https://www.youtube.com/watch?v={video['id']}"
        st.write(f"▶️ [{title}]({url})")
        if st.button(f"요약 보기 ({title})"):
            video_id = video['id']
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
                text = " ".join([t['text'] for t in transcript])
                st.text_area("📜 자막", text[:2000] + " ...")
            except Exception as e:
                st.error(f"자막 불러오기 실패: {e}")
