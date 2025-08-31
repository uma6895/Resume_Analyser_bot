import os
import gradio as gr
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ------------------------------
# 1. Load Groq API Key
# ------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY. Please set it in Hugging Face → Settings → Secrets.")

client = Groq(api_key=GROQ_API_KEY)

# ------------------------------
# 2. Embedding model
# ------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# 3. Global variables
# ------------------------------
faiss_index = None
text_chunks = []

# ------------------------------
# 4. PDF processing functions
# ------------------------------
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

def process_pdf(pdf_file):
    global faiss_index, text_chunks
    raw_text = extract_text_from_pdf(pdf_file.name)
    text_chunks = split_into_chunks(raw_text)

    if not text_chunks:
        return "⚠️ No readable text found in the PDF."

    embeddings = embedding_model.encode(text_chunks)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings))

    return "✅ PDF processed and vector index created successfully."

# ------------------------------
# 5. Query function with error handling
# ------------------------------
def query_document(query_text, top_k=3):
    if faiss_index is None or not text_chunks:
        return "⚠️ Please upload and process a PDF first."
    
    try:
        top_k = min(top_k, len(text_chunks))
        query_vector = embedding_model.encode([query_text])
        distances, indices = faiss_index.search(np.array(query_vector), top_k)
        context = "\n\n".join([text_chunks[i][:500] for i in indices[0]])

        # Groq LLM API call
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes and analyzes documents."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
            ]
        )

        # ✅ Correct access for Groq SDK
        return response.choices[0].message.content

    except Exception as e:
        print("❌ ERROR in query_document:", e)
        return f"❌ Error during query: {str(e)}"

# ------------------------------
# 6. Gradio Interface
# ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📑 PDF Q&A Bot")
    gr.Markdown("Upload a PDF, process it, and then ask questions about its content.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload your PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF")
            status_box = gr.Textbox(label="Status", interactive=False)

            question_input = gr.Textbox(label="Your Question")
            ask_btn = gr.Button("Ask")
            answer_box = gr.Textbox(label="Answer", interactive=False, lines=10)

    process_btn.click(fn=process_pdf, inputs=file_input, outputs=status_box)
    ask_btn.click(fn=query_document, inputs=question_input, outputs=answer_box)

demo.launch()
