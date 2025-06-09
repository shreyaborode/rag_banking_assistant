import streamlit as st
import json
import numpy as np
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Dynamically determine the path to embeddings.json
embeddings_path = os.path.join(os.path.dirname(__file__), "embeddings", "embeddings.json")

# Check if the embeddings file exists
if not os.path.exists(embeddings_path):
    raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}. Please run create_embeddings.py to generate it.")

# Load embeddings
with open(embeddings_path, "r") as f:
    embeddings = json.load(f)

texts = list(embeddings.keys())
vectors = np.array([v for v in embeddings.values()])
tokenizer = AutoTokenizer.from_pretrained("gpt2")# Convert to NumPy array for easier computation

# Load language model for non-RAG response
llm = pipeline("text-generation", model="gpt2",tokenizer=tokenizer,device=0)  # Replace with Gemini when available

def get_embedding(text):
    """Mock embedding function (replace with actual embedding logic if needed)."""
    return np.random.rand(len(vectors[0]))  # Replace with actual embedding logic

def compute_similarity(user_embedding):
    """Compute cosine similarity between user embedding and precomputed embeddings."""
    scores = np.dot(vectors, user_embedding) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(user_embedding)
    )
    best_idx = int(np.argmax(scores))
    return texts[best_idx], scores[best_idx]

# Streamlit UI
st.set_page_config(page_title="RAG Banking Assistant", layout="centered")
st.title("💬 RAG Banking Assistant")

user_input = st.text_input("Ask your question here:")

if st.button("Send"):
    if user_input:
        # Generate embeddings for user input
        user_embedding = get_embedding(user_input)

        # Compute RAG response
        rag_response, similarity_score = compute_similarity(user_embedding)

        # Generate non-RAG response
        non_rag_response = llm(user_input, max_length=50, num_return_sequences=1)[0]["generated_text"]

        # Display responses side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🤖 Non-RAG Response")
            st.write(non_rag_response)

        with col2:
            st.markdown("### 🧠 RAG Response")
            st.write(f"**➡️ {rag_response}**")
            st.write(f"**🔍 Similarity Score:** {similarity_score:.2f}")
    else:
        st.warning("Please enter a question.")