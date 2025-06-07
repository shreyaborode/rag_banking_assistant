from sentence_transformers import SentenceTransformer
import os
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

def read_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                # Split into chunks (e.g., paragraph-based)
                chunks = content.split('\n\n')  # double newlines = paragraph breaks
                texts.extend([chunk.strip() for chunk in chunks if chunk.strip()])
    return texts

def create_embedding(text):
    return model.encode(text).tolist()

def main():
    texts = read_texts_from_folder("data")

    embeddings = {text: create_embedding(text) for text in texts}

    with open("embeddings.json", "w", encoding='utf-8') as f:
        json.dump(embeddings, f)

    print(f"✅ {len(embeddings)} embeddings saved successfully.")

if __name__ == "__main__":
    main()
