import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json

# Load embeddings
with open("embeddings.json", "r") as f:
    embeddings = json.load(f)

texts = list(embeddings.keys())
vectors = [v for v in embeddings.values()]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.set_page_config(page_title="RAG Banking Assistant", layout="centered")
st.title("💬 RAG Banking Assistant")

user_input = st.text_input("Ask your question here:")

if st.button("Send"):
    if user_input:
        user_embedding = model.encode(user_input)
        scores = util.cos_sim(user_embedding, vectors)[0]
        best_score = float(scores.max())
        best_idx = int(scores.argmax())

        st.markdown("### 🧠 Best Match")
        st.write(f"**➡️ {texts[best_idx]}**")
        st.write(f"**🔍 Similarity Score:** {best_score:.2f}")
    else:
        st.warning("Please enter a question.")
