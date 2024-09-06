from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.data_ingestion import ingestion
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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



model = ChatGoogleGenerativeAI(model="gemini-1.0-pro",convert_system_message_to_human=True)


def retrieve_generate():
    chunks_path = ingestion()
    vector_store = Chroma(persist_directory= chunks_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever()
    # chat_prompt = ChatPromptTemplate(
    # [
    #     ("system", system_prompt),
    #     ("human", "{input}")
    # ])

    # question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
    # rag_chain= create_retrieval_chain(retriever, question_answering_chain)
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )
    return conversational_rag_chain


if __name__ == "__main__":
   chain= retrieve_generate()
   answer= chain.invoke(
    {"input": "Can give the summary of reviews from the students"},
    config={"configurable": {"session_id": "abc123"}},
    )["answer"]
   print(answer)