import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

# HuggingFace Token (required for MiniLM)
os.environ['HF_TOKEN'] = "your_huggingface_token"  # Optional for public models

st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("🧠 Conversational RAG with PDF Uploads")
st.markdown("Upload PDFs and ask questions using **Groq's Gemma-2-9B-IT**. Contextual chat memory enabled!")

# Get Groq API Key
api_key = st.text_input("🔐 Enter your Groq API Key", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma-2-9b-it")

    session_id = st.text_input("🗂️ Session ID", value="default")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("📤 Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            documents.extend(docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and a follow-up question, rewrite it as a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        retriever_with_context = create_history_aware_retriever(llm, retriever, contextualize_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the following context to answer the question. Be concise and honest.\\n\\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever_with_context, qa_chain)

        def get_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        def is_relevant(question, retriever):
            return bool(retriever.get_relevant_documents(question))

        user_input = st.text_input("💬 Ask a question related to the PDF")

        if user_input:
            if not is_relevant(user_input, retriever):
                st.warning("❌ Your question does not match any content in the PDF.")
            else:
                response = conversational_rag.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.success(response["answer"])
else:
    st.info("🔑 Enter your **Groq API key** above to start.")
