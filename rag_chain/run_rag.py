import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the same model used for creating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_embeddings(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_user_query_embedding(query):
    return model.encode(query).reshape(1, -1)

def find_most_similar(query_embedding, embeddings_dict):
    texts = list(embeddings_dict.keys())
    vectors = np.array(list(embeddings_dict.values()))
    
    similarities = cosine_similarity(query_embedding, vectors)
    best_idx = np.argmax(similarities)
    
    return texts[best_idx], similarities[0][best_idx]

def main():
    embeddings = load_embeddings("embeddings.json")
    
    query = input("Ask your question: ")
    query_embedding = get_user_query_embedding(query)
    
    best_match, score = find_most_similar(query_embedding, embeddings)
    
    print(f"\n🧠 Best match:\n➡️ \"{best_match}\"\n🔍 Similarity score: {score:.2f}")

if __name__ == "__main__":
    main()
