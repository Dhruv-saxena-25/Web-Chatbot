from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from src.retrieval_generation import retrieve_generate
from src.data_ingestion import ingestion


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-1.0-pro",convert_system_message_to_human=True)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question "
    "If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

retriever_prompt = ("Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
chat_history= []

store = {}
def get_session_history(session_id: str)-> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id]= ChatMessageHistory()
  return store[session_id]




app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form["msg"]
    input = msg
    chain = retrieve_generate()
    result= chain.invoke(
        {"input": input},
        config={"configurable": {"session_id": "abc123"}},
        )["answer"]
    print("Response : ", result)
    return str(result)

if __name__ == '__main__':
    app.run(debug= True, host="0.0.0.0", port=8080)

