import os
import sys
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber

# --------------------------
# Configuration
# --------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

INDEX_FILE = "rag_index.faiss"
META_FILE = "rag_meta.pkl"
BATCH_SIZE = 32  # Number of chunks to embed at once

# --------------------------
# Load / Save FAISS index
# --------------------------
def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    else:
        d = embedding_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(d)
        return index, []


def save_index(index, metadata):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

# --------------------------
# Text extraction
# --------------------------
def extract_text_from_file(filepath):
    """Extract text from TXT or PDF. Ignores images/tables."""
    text = ""
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    elif filepath.endswith(".pdf"):
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    print(f"⚠️ Page {page_num} has no extractable text, skipping.")
    return text

# --------------------------
# Add book to RAG store
# --------------------------
def add_to_store(raw_text: str, book_name: str = "uploaded_book"):
    index, metadata = load_index()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    if not chunks:
        print("❌ No chunks extracted from the book.")
        return False

    print(f"📚 Book split into {len(chunks)} chunks. Starting embeddings...")

    # Batch embedding
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        embeddings = embedding_model.encode(batch_chunks, show_progress_bar=False)
        index.add(embeddings)
        metadata.extend([(book_name, c) for c in batch_chunks])
        print(f"✅ Processed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks...")

    save_index(index, metadata)
    print(f"🎉 Finished storing book '{book_name}' with {len(chunks)} chunks.")
    return True

# --------------------------
# Query RAG store
# --------------------------
def query_store(query: str, top_k: int = 5):
    index, metadata = load_index()
    if index.ntotal == 0:
        return ["Knowledge base is empty. Please upload a book."]

    q_emb = embedding_model.encode([query])
    distances, indices = index.search(q_emb, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx][1])
    return results

# --------------------------
# CLI for preloading books
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_store.py <book_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"📖 Loading book: {filepath}")
    raw_text = extract_text_from_file(filepath)

    add_to_store(raw_text, book_name=os.path.basename(filepath))
