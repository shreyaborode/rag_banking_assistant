import pandas as pd

def load_smartcheck_excel(file_path):
    df = pd.read_excel(file_path)
    docs = []
    for _, row in df.iterrows():
        docs.append(f"Rule ID: {row['Rule ID']}\nRule Name: {row['Rule Name']}\nDescription: {row['Rule Description']}")
    return docs

def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [file.read()]
