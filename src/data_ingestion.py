from langchain.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.vectordb import vector_db
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
URLs = [
    "https://zepanalytics.com/",
    "https://zepanalytics.com/courses",
    "https://zepanalytics.com/bundle",
    "https://zepanalytics.com/projects",
    "https://zepanalytics.com/blogs",
    "https://zepanalytics.com/virtual-internship",
    "https://zepanalytics.com/courses/python-programming-for-data-science-a-z",
    "https://zepanalytics.com/courses/microsoft-power-bi-a-complete-guide-2023-edition",
    "https://zepanalytics.com/courses/cnn-everything-about-convolution-neural-networks",
    
        ]

def ingestion():
    loder = UnstructuredURLLoader(urls= URLs)
    data = loder.load()
    text_splitter = RecursiveCharacterTextSplitter(separators= '\n',
                                              chunk_size=1000, 
                                              chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    chunks_path = vector_db(data=chunks, embeddings=embeddings)
    return chunks_path


if __name__ == "__main__":
    
    chunks_path = ingestion()
    print(chunks_path)






 