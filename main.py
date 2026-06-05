from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

print("Loading model and index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("sfa_index.faiss")

with open("sfa_chunks.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
meta = data["meta"]
print(f"Ready! {len(chunks)} chunks loaded")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    q_embed = model.encode([q.question]).astype('float32')
    D, I = index.search(q_embed, k=3)
    
    context = "\n\n".join([
        f"[Source: {meta[i]}]\n{chunks[i]}"
        for i in I[0] if i < len(chunks)
    ])
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=f'You are a Safe Food Alliance assistant. Answer ONLY using this context. If the answer is not in the context say I dont have that information.\n\nContext:\n{context}',
        messages=[
            {"role": "user", "content": q.question}
        ]
    )
    
    sources = list(set([meta[i] for i in I[0] if i < len(chunks)]))
    
    return {
        "answer": response.content[0].text,
        "sources": sources
    }

@app.get("/")
def home():
    return FileResponse("index.html")