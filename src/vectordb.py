from langchain_chroma import Chroma
from datetime import datetime
import os

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def vector_db(documents, embeddings):
    base_dir = os.makedirs("database", exist_ok=True)
    vectors_path = os.path.join(base_dir, timestamp)
    Chroma.from_documents(documents, embeddings, persist_directory= vectors_path)
    return vectors_path







