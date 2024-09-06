from langchain_chroma import Chroma
from datetime import datetime
import os


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def vector_db(data, embeddings):
    os.makedirs("database", exist_ok=True)
    collection_name = f"{timestamp}/chroma/"
    vectors_path = os.path.join("database", collection_name)
    Chroma.from_documents(data, embeddings, persist_directory= vectors_path)
    return vectors_path








