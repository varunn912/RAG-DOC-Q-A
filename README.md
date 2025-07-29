📌 Overview
This project implements a RAG-based Document Question Answering System that combines state-of-the-art language models with document retrieval techniques. Inspired by architectures like the Transformer and models such as LLaMA, PaLM, and T5, it leverages attention mechanisms and pre-trained LLMs to retrieve relevant passages from your documents and generate precise answers.

The system supports:

Uploading and parsing .pdf files
Semantic chunking and indexing using embeddings
Efficient similarity search via vector databases
Natural language question answering using LLMs
Auto-regressive, context-aware generation (as seen in models like Transformer)
🔍 Based on principles from: "Attention Is All You Need" (Vaswani et al.) and modern LLM advancements (e.g., LLaMA, GPT, T5) 

🧩 Features
✅ Document Upload & Parsing
Supports uploading .pdf files (e.g., Attention.pdf, LLM.pdf) for analysis.

✅ Retrieval-Augmented Generation (RAG)

Chunks document content into meaningful segments
Encodes chunks into dense vectors using embedding models
Retrieves top-k relevant contexts when a query is made
Feeds retrieved context + query to an LLM for answer generation
✅ Transformer-Based Architecture
Leverages self-attention mechanisms for contextual understanding, similar to the Transformer model described in the Attention.pdf.

✅ Integration with Modern LLMs
Compatible with open-source LLMs such as:

LLaMA / LLaMA-2
T5 / mT5
StarCoder / CodeGen
BLOOM, OPT, etc.
✅ Flexible & Extensible
Designed for easy integration with:

Vector stores (e.g., FAISS, Chroma, Pinecone)
Hugging Face Transformers
LangChain or LlamaIndex frameworks
