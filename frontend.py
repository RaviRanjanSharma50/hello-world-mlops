import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/search"

# Streamlit page config
st.set_page_config(page_title="Hybrid RAG Frontend", page_icon="🤖")

st.title("🤖 Hybrid RAG Search")
st.write("Enter a query to search across BM25, Vector, Keyword, fused with RRF, reranked by BGE.")

# Input box
query = st.text_input("Enter your query:")

# Slider for number of results
k = st.slider("Number of results", min_value=1, max_value=10, value=5)

# Submit button
if st.button("Search"):
    if query.strip():
        try:
            response = requests.post(API_URL, json={"query": query, "k": k})
            if response.status_code == 200:
                results = response.json().get("results", [])
                st.success("Top Results:")
                for i, res in enumerate(results, 1):
                    st.markdown(f"**Result {i}:**\n{res}\n---")
            else:
                st.error(f"Backend error: {response.status_code}")
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")
    else:
        st.warning("Please enter a query before searching.")
