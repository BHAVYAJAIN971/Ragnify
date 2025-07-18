# 🧠 Ragnify – Offline RAG-based AI Document Assistant

**Ragnify** is a powerful **RAG (Retrieval Augmented Generation)** based offline AI tool built to deliver **accurate, document-grounded answers** from your PDF files. Designed for **complete offline use**, it integrates **OCR capabilities** and **LLM (LLaMA 2 7B GGUF)** models to provide highly relevant, hallucination-free answers.

> 🔧 Developed by **Bhavya Jain** and **Aryan Singh**  
> 🧑‍🏫 Guided by **Mr. Ankit Pratap Sir** during our internship at **Tech Mahindra**  

---

## 🧩 Key Features

- 🔍 **RAG-based Context Retrieval** — Ensures answers are backed by the document.
- 🧠 **LLaMA 2 Integration (Offline)** — Uses 7B GGUF model with no external calls.
- 📥 **Multi-PDF Upload** — Seamlessly handle multiple documents.
- 🧾 **OCR Support with Tesseract** — Converts scanned or image-based PDFs to text.
- 🔐 **100% Offline** — No data sent to the cloud, perfect for private and secure use.
- 🚫 **Zero Hallucinations** — Model only answers based on available context.
- 🎯 **Fast and Accurate** — High performance even on limited hardware.
- 🪄 **Minimal UI** — Built with Streamlit for simplicity and speed.
- 🛠️ **Customizable** — Easily adapt to new document sets or other LLMs.

---

## 🔍 How It Works

1. **PDF Upload**: Users upload one or more PDFs via the Streamlit interface.
2. **OCR Processing**: Tesseract automatically extracts text from scanned/image PDFs.
3. **Chunking**: Documents are broken down into manageable text chunks.
4. **Embedding**: Each chunk is embedded and stored in a **FAISS vector database**.
5. **Query Input**: User asks a question through the UI.
6. **Context Retrieval**: Semantic search fetches relevant chunks.
7. **Answer Generation**: The **LLaMA 2 model** generates an accurate, context-aware answer.

---

## 🧠 What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI technique where external knowledge (PDFs in our case) is retrieved and used to **augment the input to the language model**, ensuring more accurate and grounded responses.

---

## 🗃️ Project Structure

```bash
Ragnify/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── llm_model/              # Folder for GGUF LLaMA model
├── ocr_utils/              # OCR setup and utilities (Tesseract)
├── utils/                  # PDF processing, chunking, embedding
├── faiss_index/            # Vector store (FAISS)
├── stored_pdfs/            # Uploaded PDF storage
└── README.md               # You're here!
🧪 Future Improvements
🌐 Multi-language support for PDFs

☁️ Optional cloud backup or API-based LLM extensions

📊 Visualization support for data-heavy documents (tables, charts)

⚙️ Docker setup for cross-platform ease of deployment

🔄 Add CSV/Word support in addition to PDF

🤖 Add voice-based query input and response

🧬 Fine-tune LLM for legal, medical, or research domains

💬 Contact
📧 Email: bhavyajain7900@gmail.com
linkedin:https://www.linkedin.com/in/bhavya-jain-b90960245/
