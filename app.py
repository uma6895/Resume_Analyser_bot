import os
import gradio as gr
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# 2. Groq API Setup (use environment variable in production)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")  # Replace "your_api_key_here" for local testing
client = Groq(api_key=GROQ_API_KEY)

# 3. Helper functions
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables to hold index and chunks
faiss_index = None
text_chunks = []

def process_pdf(pdf_file):
    global faiss_index, text_chunks
    raw_text = extract_text_from_pdf(pdf_file.name)
    text_chunks = split_into_chunks(raw_text)
    embeddings = embedding_model.encode(text_chunks)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings))
    return "PDF processed and vector index created successfully."

def query_document(query_text, top_k=3):
    if faiss_index is None or not text_chunks:
        return "Please upload and process a PDF first."
    
    query_vector = embedding_model.encode([query_text])
    distances, indices = faiss_index.search(np.array(query_vector), top_k)
    context = "\n\n".join([text_chunks[i] for i in indices[0]])

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are an assistant that summarizes and analyzes documents."},
            {"role": "user", "content": f"{context}\n\nQuestion: {query_text}"}
        ]
    )
    return response.choices[0].message.content


iface = gr.Interface(
    fn=lambda file, question: (
        process_pdf(file),         # status output
        query_document(question)   # answer output
    ),
    inputs=[
        gr.File(label="Upload your PDF"),
        gr.Textbox(label="Enter your question")
    ],
    outputs=[
        gr.Textbox(label="Status", interactive=False),
        gr.Textbox(label="Answer", interactive=False)
    ],
    title="Resume Q&A Bot",
    description="Upload a PDF and ask questions about its content."
)

iface.launch()

