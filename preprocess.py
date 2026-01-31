# -----------------------------------------------------------------------------
# Preprocess Company_Brochure.pdf into engagepro.pkl
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
import PyPDF2
from config import hf_embeddings

def load_pdf_text(pdf_path):
    print(f"Loading PDF: {pdf_path}")
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        print(f"Total pages: {len(reader.pages)}")
        # Adding a separator between pages to help the LLM identify page breaks
        for i, page in enumerate(reader.pages):
            text += f"\n--- Page {i+1} ---\n"
            text += page.extract_text()
    print(f"Extracted {len(text)} characters")
    return text

# We use a much larger chunk size (2000) and overlap (500) 
# to ensure values split across pages are captured in a single context window.
def chunk_text(text, chunk_size=2000, overlap=500):
    print("Chunking text with high overlap...")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(text):
            break
    print(f"Created {len(chunks)} chunks")
    return chunks

def compute_embeddings(chunks):
    print(f"Computing embeddings for {len(chunks)} chunks...")
    embeddings = hf_embeddings.encode(chunks, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")
    return np.array(embeddings)

def save_to_pickle(chunks, embeddings, output_path="engagepro.pkl"):
    print("Saving to pickle...")
    df = pd.DataFrame({
        "embedding": [embeddings[i] for i in range(len(embeddings))],
        "source": ["Company_Brochure.pdf"] * len(chunks),
        "content": chunks,
    })
    df.to_pickle(output_path)
    print(f"Saved to {output_path}")

print("Starting preprocessing...")
pdf_path = "Company_Brochure.pdf"
text = load_pdf_text(pdf_path)

# Parameters optimized to prevent "Core Values" being split
chunks = chunk_text(text, chunk_size=2000, overlap=500)

embeddings = compute_embeddings(chunks)
save_to_pickle(chunks, embeddings)
print("Done!")
