# LangChain-Document-Based-Q-A-System

📄 Document-based Q&A System (Hugging Face + FAISS + Streamlit)
This is a Generative AI powered web application that allows users to upload documents (PDF or Word files) and ask questions based on their content. The system uses Hugging Face's open-source models, vector similarity search, and LLMs to return intelligent answers grounded in the uploaded documents.

Screenshots:
<img width="987" height="767" alt="image" src="https://github.com/user-attachments/assets/568d2709-47ab-4dcf-855a-cc69c53b3248" />
<img width="975" height="595" alt="image" src="https://github.com/user-attachments/assets/e1dbac71-d4c6-4936-83b7-e01c6ae61e4b" />


🎯 Key Features
📎 Upload support for PDF and Word (DOCX) files

📚 Text chunking and embedding using sentence-transformers

🧠 Vector store creation with FAISS for efficient similarity search

💬 LLM-powered question answering using Hugging Face's Mixtral-8x7B model

🖼️ Simple, intuitive Streamlit UI

✅ Fully local embedding and retrieval, with cloud-based LLM inference

🛠️ Tech Stack
Component	Library/Tool
Frontend	Streamlit
LLM	Hugging Face Endpoint (Mixtral)
Embeddings	SentenceTransformers (MiniLM)
Vector Store	FAISS
File Parsing	PyMuPDF (PDF) & python-docx (Word)
Chunking	LangChain Text Splitters
Environment	Python (.env for API tokens)

📂 How it Works
File Upload: User uploads a .pdf or .docx file via the UI.

Text Extraction: Text is extracted using PyMuPDF for PDFs and python-docx for Word files.

Chunking: Extracted text is broken into smaller overlapping chunks.

Embedding: Each chunk is embedded using all-MiniLM-L6-v2 from sentence-transformers.

Vector Store: The chunks are stored in a FAISS vector store.

User Query: A user types a question in the UI.

Retrieval: Top relevant chunks are retrieved using vector similarity.

Answer Generation: These chunks + the question are passed to the Hugging Face LLM (e.g., Mixtral) to generate a response.

⚠️ Performance Note
This app uses the mistralai/Mixtral-8x7B-Instruct-v0.1 hosted on Hugging Face's inference endpoint.
⚠️ The initial response may be slow due to model loading time and API throttling on the free tier.

💡 For faster responses and better latency, you are encouraged to use ChatOpenAI (gpt-3.5 / gpt-4) via OpenAI API by swapping the backend LLM. LangChain supports easy integration with ChatOpenAI.

⚠️ Setup Instructions
✅ Prerequisites
Python 3.9–3.12

A Hugging Face account with an API token

Internet access for the LLM via Hugging Face Inference API

<img width="1127" height="739" alt="image" src="https://github.com/user-attachments/assets/a4236cf7-0cb0-43ea-a464-2303954ba5d7" />
<img width="1091" height="555" alt="image" src="https://github.com/user-attachments/assets/18cef020-efc4-4ddf-9a8a-88cf4c599ee4" />


🧪 Example Use Cases
📚 Students asking questions from study material

🧑‍⚖️ Lawyers reviewing lengthy legal documents

🏢 Enterprise teams searching knowledge base PDFs

📘 Techniques Used
You can select from the dropdown in the UI to view the technique used:

SentenceTransformers (MiniLM) for embeddings

FAISS for vector store

Hugging Face LLM (Mixtral-8x7B-Instruct) via API

LangChain chaining for prompt templates and orchestration

🔮 Future Prospects
✅ Add ChatOpenAI support for faster and cheaper LLM interaction

📌 Add support for multi-document ingestion and ranking

🧠 Integrate RAG (Retrieval-Augmented Generation) pipelines using LangChain Expression Language

🗃️ Add persistent vector storage using ChromaDB, Qdrant, or Pinecone

📊 Display source references or citations along with answers

🌐 Deploy on platforms like Streamlit Cloud, Hugging Face Spaces, or AWS EC2

📱 Extend to a mobile-friendly UI using Streamlit or Gradio


❤️ Attribution
Made with ❤️ by Shivendra using Hugging Face, FAISS, and Streamlit.


