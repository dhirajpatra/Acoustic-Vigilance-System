# chat_service/app.py

import streamlit as st
import requests
import os
from io import BytesIO

# FastAPI endpoints
RAG_UPLOAD_URL = "http://rag_service:8000/upload_rag"
COMPARE_URL = "http://comparison_service:8001/compare"

st.title("Acoustic Vigilance Chat")
st.markdown("Upload reference sounds or test machine audio for analysis.")

# Tab layout
tab1, tab2 = st.tabs(["RAG Upload", "Sound Check"])

with tab1:
    st.subheader("Upload Reference Sound (RAG)")
    rag_file = st.file_uploader("Choose a reference audio file (e.g., good machine sound)", type=["wav", "mp3"])
    if rag_file and st.button("Upload to RAG"):
        files = {"file": (rag_file.name, rag_file.getvalue())}
        response = requests.post(RAG_UPLOAD_URL, files=files)
        st.success(f"Uploaded! Status: {response.json()['status']}")

with tab2:
    st.subheader("Test Machine Sound")
    test_file = st.file_uploader("Upload a test audio clip (30 sec recommended)", type=["wav", "mp3"])
    if test_file and st.button("Analyze"):
        files = {"file": (test_file.name, test_file.getvalue())}
        response = requests.post(COMPARE_URL, files=files)
        result = response.json()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Similarity Score", f"{result['similarity']:.2f}")
        with col2:
            status = "✅ Healthy" if result["diagnosis"] == "healthy" else "⚠️ Faulty"
            st.metric("Diagnosis", status)
        
        # Optional: Show audio player
        st.audio(test_file, format=f"audio/{test_file.name.split('.')[-1]}")