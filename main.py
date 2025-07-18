import os, fitz, base64, pytesseract, requests
import streamlit as st
import numpy as np
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from streamlit_lottie import st_lottie
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# === CONFIG ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
UPLOAD_DIR, HISTORY_DIR = "stored_pdfs", "chat_history"
MODEL_PATH = os.path.abspath("./models/llama-2-7b-chat.Q4_K_M.gguf")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
st.set_page_config(layout="wide", page_title="Ragnify chat bot")

# === LOTTIE ===
def load_lottie_url(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

with st.container():
    st_lottie(load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_0yfsb3a1.json"), height=200, key="header")

# === F1 THEME STYLE ===
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #111;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #e10600;
        color: white;
        border-radius: 10px;
        border: none;
    }
    .stTextInput>div>div>input {
        border: 2px solid #e10600;
        border-radius: 5px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #e10600;
    }
    </style>
""", unsafe_allow_html=True)

# === LOAD MODELS ===
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        max_tokens=512,
        temperature=0.7,
        n_gpu_layers=20,
        model_kwargs={"model_type": "llama"},
        verbose=False,
    )

embed, llm = get_embedder(), get_llm()

# === SIDEBAR FILE UPLOAD ===
st.sidebar.title("📤 Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs (up to 1GB)", type="pdf", accept_multiple_files=True
)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.read())
    st.sidebar.success("✅ Upload complete")

pdf_files = os.listdir(UPLOAD_DIR)
selected_pdfs = st.sidebar.multiselect("📑 Select PDFs", pdf_files, default=pdf_files[:2])
pdf_paths = [os.path.join(UPLOAD_DIR, f) for f in selected_pdfs if os.path.exists(os.path.join(UPLOAD_DIR, f))]

# === PARSE PDFs ===
def parse_pdf(path):
    chunks, texts = [], []
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                if not text.strip():
                    img = convert_from_path(path, first_page=i+1, last_page=i+1)[0]
                    text = pytesseract.image_to_string(img)
                if text.strip():
                    texts.append((i + 1, text))
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        for pg, content in texts:
            for chunk in splitter.split_text(content):
                chunks.append(Document(page_content=chunk, metadata={"page": pg}))
        return chunks
    except Exception as e:
        st.error(f"❌ Error parsing {path}: {e}")
        return []

all_docs = []
for path in pdf_paths:
    all_docs += parse_pdf(path)

if all_docs:
    embeddings = embed.embed_documents([doc.page_content for doc in all_docs])
    faiss_db = FAISS.from_documents(all_docs, embedding=embed)

# === VIEW PDFs SIDE-BY-SIDE ===
st.subheader("📖 View PDFs")
search_term = st.text_input("🔍 Highlight keyword across all PDFs")
cols = st.columns(len(pdf_paths))
for idx, path in enumerate(pdf_paths):
    with cols[idx]:
        with fitz.open(path) as doc:
            temp = f"temp_{idx}.pdf"
            highlighted = False
            for i, page in enumerate(doc):
                if search_term:
                    boxes = page.search_for(search_term)
                    for box in boxes:
                        page.add_highlight_annot(box)
                        highlighted = True
                    if highlighted:
                        break
            doc.save(temp)
            with open(temp, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{encoded}#page={i+1 if highlighted else 1}" '
                f'width="100%" height="450" type="application/pdf"></iframe>',
                unsafe_allow_html=True
            )
            st.caption(f"📄 {os.path.basename(path)}")

# === Q&A ===
st.markdown("---")
query = st.text_input("💬 Ask a question across selected PDFs")
if query and all_docs:
    retriever = faiss_db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    result = qa.invoke({"query": query})
    st.markdown("### ✅ LLaMA Answer")
    st.success(result["result"])
    for doc in result["source_documents"]:
        st.markdown(f"📄 Page: {doc.metadata.get('page', 'N/A')}")
        st.markdown(f"> {doc.page_content[:300]}...")

    with open(os.path.join(HISTORY_DIR, "chat_history.txt"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()}\nQ: {query}\nA: {result['result']}\n\n")

# === ML MODEL COMPARISON ===
if query and embeddings:
    st.markdown("### 🧠 Compare Answer Scores from ML Models")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": xgb.XGBRegressor()
    }
    q_vec = embed.embed_query(query)
    x_train, y_train = np.array(embeddings), np.dot(embeddings, q_vec)

    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            pred = model.predict(x_train)
            top_idx = int(np.argmax(pred))
            best_chunk = all_docs[top_idx]
            st.subheader(f"🔍 {name}")
            st.write(best_chunk.page_content)
            st.caption(f"📄 Page: {best_chunk.metadata.get('page', 'N/A')}")
            st.markdown("---")
        except Exception as e:
            st.warning(f"{name} failed: {e}")

# === DOWNLOAD HISTORY ===
if os.path.exists(os.path.join(HISTORY_DIR, "chat_history.txt")):
    with open(os.path.join(HISTORY_DIR, "chat_history.txt"), "r", encoding="utf-8") as f:
        st.download_button("📥 Download Chat History", f.read(), file_name="chat_history.txt")
