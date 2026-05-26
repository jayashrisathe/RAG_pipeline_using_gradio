import torch
import transformers
import pypdf
import os
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

def print_markdown(text):
    """Prints text (Markdown display only works in Jupyter/Colab, plain print used here)."""
    print(text)

# ── PDF Reader ─────────────────────────────────────────────
def read_pdf(filename):
    """Read text from any PDF uploaded to the workspace."""
    workspace_path = os.path.join(os.path.dirname(__file__), filename)
    path = workspace_path if os.path.exists(workspace_path) else filename

    if not os.path.exists(path):
        print(f"File not found: {filename}")
        print("Available PDFs:", [f for f in os.listdir(os.path.dirname(__file__) or ".") if f.endswith(".pdf")])
        return ""

    print(f"Reading {filename}...")
    reader = pypdf.PdfReader(path)
    print(f"Pages: {len(reader.pages)}")

    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_text.append(text)

    full_text = "\n".join(pages_text)
    print(f"Extracted {len(full_text)} characters")
    return full_text

def read_pdf_preview(filename, chars=500):
    """Read and print the first `chars` characters of a PDF."""
    text = read_pdf(filename)
    if text:
        print("\n--- Preview ---")
        print(text[:chars])
        print("...")

def list_uploaded_pdfs():
    folder = os.path.dirname(__file__) or "."
    pdfs = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if pdfs:
        print("Uploaded PDFs:", pdfs)
    else:
        print("No PDFs found. Use the Upload File button to add one.")
    return pdfs

# ── Tokenizer ──────────────────────────────────────────────
def get_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer("Hello, let's join sync up call.")
    print("tokens==========", tokens)
    print("word tokens:", tokenizer.tokenize("Hello, let's join sync up call."))
    return tokenizer

# ── Sentiment Analysis ─────────────────────────────────────
def get_transformer_score(text="Apple lost 10 Million dollars today due to US tariffs"):
    pipe = transformers.pipeline(model="ProsusAI/finbert", device_map="cpu")
    output = pipe(text)
    print("output==========", output)
    return output

# ── PDF Question Answering ─────────────────────────────────
def bits_byte():
    try:
        pdfs = list_uploaded_pdfs()
        if not pdfs:
            print("No PDFs found. Exiting.")
            return

        context = read_pdf(pdfs[0])
        question = "What is price of apple per kg?"
        prompt_template = f"""<|system|>
    You are an AI assistant. Answer the following question based *only* on the provided document text. If the answer is not found in the document, say "The document does not contain information on this topic." Do not use any prior knowledge.

    Document Text:
    ---
    {context}
    ---
    <|end|>
    <|user|>
    Question: {question}<|end|>
    <|assistant|>
    Answer:"""

        print(f"\nContext (first 200 chars): {context[:200]}...")
        print(f"Question: {question}\n")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
        qa_pipe = transformers.pipeline("text-generation",
                model="microsoft/Phi-4-mini-instruct",
                tokenizer=tokenizer,
                torch_dtype="auto",
                device_map="auto"
                )
        result = qa_pipe(prompt_template,
                           max_new_tokens=500,
                           do_sample=True,
                           temperature=0.2,
                           top_p=0.9)
        print("result======", result)

    except Exception as e:
        import traceback
        traceback.print_exc()


pdfs = list_uploaded_pdfs()
if pdfs:
    read_pdf_preview(pdfs[0])

bits_byte()
