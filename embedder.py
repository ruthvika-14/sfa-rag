from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from scraper import scrape_all

def chunk_text(text, size=200, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_and_save():
    print("Scraping pages...")
    docs = scrape_all()
    
    all_chunks = []
    all_meta = []
    
    for doc in docs:
        chunks = chunk_text(doc['text'])
        for c in chunks:
            all_chunks.append(c)
            all_meta.append(doc['url'])
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Creating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, "sfa_index.faiss")
    with open("sfa_chunks.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "meta": all_meta}, f)
    
    print(f"Done! Index saved with {len(all_chunks)} chunks")

if __name__ == "__main__":
    build_and_save()