import fitz  # PyMuPDF
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# FIX 1: Load embedding model once at module level
model = SentenceTransformer('all-MiniLM-L6-v2')

DATA_PATH = "Research_papers/"
DB_PATH = "db/index.faiss"
CHUNKS_PATH = "db/chunks.txt"


def extract_text(pdf_path):
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# FIX 2: Added overlap to prevent sentences from being cut at boundaries
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.
    overlap=50 means 50 words from previous chunk are repeated
    at the start of next chunk — preserves context at boundaries.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap  # how far to advance each time

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# FIX 3: normalize_embeddings=True for cosine similarity compatibility
def get_embedding(text):
    """Generate normalized embedding for cosine similarity."""
    return model.encode(text, normalize_embeddings=True).astype("float32")


def main():
    all_chunks = []
    embeddings = []

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {DATA_PATH}")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting ingestion...\n")

    for file in pdf_files:
        path = os.path.join(DATA_PATH, file)
        print(f"Processing: {file}")

        try:
            text = extract_text(path)

            if not text.strip():
                print(f"No text extracted from {file}, skipping.")
                continue

            chunks = chunk_text(text)
            print(f"  → {len(chunks)} chunks created")

            for chunk in chunks:
                emb = get_embedding(chunk)
                embeddings.append(emb)
                # Tag each chunk with its source filename for filtering later
                all_chunks.append(f"[{file}] {chunk}")

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    if not embeddings:
        print("No embeddings generated. Check your PDF files.")
        return

    # Build FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)  # L2 on normalized vectors = cosine similarity
    index.add(np.array(embeddings))

    # Save index and chunks
    os.makedirs("db", exist_ok=True)
    faiss.write_index(index, DB_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.replace("\n", " ") + "\n---\n")

    print(f"\nIngestion complete!")
    print(f"   Total chunks stored : {len(all_chunks)}")
    print(f"   FAISS index saved   : {DB_PATH}")
    print(f"   Chunks saved        : {CHUNKS_PATH}")


if __name__ == "__main__":
    main()