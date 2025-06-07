import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
