from sentence_transformers import SentenceTransformer
import os
import json
import pandas as pd

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Open-source model

def read_texts_from_txt(file_path):
    """Read text from a .txt file and split into chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Split into chunks (e.g., paragraph-based)
        chunks = content.split('\n\n')  # Double newlines = paragraph breaks
        return [chunk.strip() for chunk in chunks if chunk.strip()]

def read_texts_from_excel(file_path):
    """Read text from an Excel file."""
    df = pd.read_excel(file_path)
    # Combine all columns into a single list of strings
    return df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()

def create_embedding(text):
    """Generate embeddings for a given text."""
    return model.encode(text).tolist()

def main():
    """Main function to create embeddings."""
    folder_path = "data"
    texts = []

    # Process SQL queries
    sql_file = os.path.join(folder_path, "sql_queries.txt")
    if os.path.exists(sql_file):
        texts.extend(read_texts_from_txt(sql_file))

    # Process UI components
    ui_file = os.path.join(folder_path, "ui_components.txt")
    if os.path.exists(ui_file):
        texts.extend(read_texts_from_txt(ui_file))

    # Process Smartcheck rules (Excel)
    smartcheck_file = os.path.join(folder_path, "smartcheck_rules.xlsx")
    if os.path.exists(smartcheck_file):
        texts.extend(read_texts_from_excel(smartcheck_file))

    # Generate embeddings
    embeddings = {text: create_embedding(text) for text in texts}

    # Save embeddings to a JSON file
    with open("embeddings.json", "w", encoding='utf-8') as f:
        json.dump(embeddings, f)

    print(f"✅ {len(embeddings)} embeddings saved successfully.")

if __name__ == "__main__":
    main()