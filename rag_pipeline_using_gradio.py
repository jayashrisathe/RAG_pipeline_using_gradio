from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# ── Config ─────────────────────────────────────────────────
DATA_FILE_PATH = "/content/sample_data/sample.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ── 1. Load & Split Documents ─────────────────────────────
print(f"Attempting to load data from: {DATA_FILE_PATH}")
loader = TextLoader(DATA_FILE_PATH, encoding="utf-8")
raw_documents = loader.load()
print(f"Successfully loaded {len(raw_documents)} document(s).")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
documents = text_splitter.split_documents(raw_documents)

if not documents:
    raise ValueError("Error: Splitting resulted in zero documents. Check the input file and splitter settings.")
print(f"Document split into {len(documents)} chunks.")

# ── 2. Create Embeddings & Vector Store ────────────────────
print(f"Initializing HuggingFace Embeddings model '{EMBEDDING_MODEL}'...")
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

huggingface_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Creating ChromaDB vector store...")
vector_store = Chroma.from_documents(documents=documents, embedding=huggingface_embeddings)
vector_count = vector_store._collection.count()
print(f"ChromaDB vector store created with {vector_count} items.")

if vector_count == 0:
    raise ValueError("Vector store creation resulted in 0 items. Check previous steps.")

# ── 3. Load LLM ───────────────────────────────────────────
print(f"Loading LLM: {LLM_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype="auto", device_map="auto")
pipe = hf_pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
    max_length=4096,
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=pipe)
print("Model loaded successfully!")

# ── 4. Build QA Chain ──────────────────────────────────────
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)
print("RetrievalQAWithSourcesChain created.")


def ask_questions(query):
    result = qa_chain.invoke({"question": query})
    print(f"\nAnswer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    return result["answer"], result["sources"]


# ── 5. Gradio UI ──────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Eleven Madison Park Q&A Assistant") as demo:
    gr.Markdown(
        """
        # Eleven Madison Park - AI Q&A Assistant 💬
        Ask questions about the restaurant based on its website data.
        The AI provides answers and cites the source document.
        *(Examples: What are the menu prices? Who is the chef? Is it plant-based?)*
        """
    )

    question_input = gr.Textbox(
        label="Your Question:",
        placeholder="e.g., What are the opening hours on Saturday?",
        lines=2,
    )

    with gr.Row():
        answer_output = gr.Textbox(label="Answer:", interactive=False, lines=6)
        sources_output = gr.Textbox(label="Sources:", interactive=False, lines=2)

    with gr.Row():
        submit_button = gr.Button("Ask Question", variant="primary")
        clear_button = gr.ClearButton(
            components=[question_input, answer_output, sources_output],
            value="Clear All"
        )

    gr.Examples(
        examples=[
            "What are the different menu options and prices?",
            "Who is the head chef?",
            "What is Magic Farms?",
        ],
        inputs=question_input,
        cache_examples=False,
    )

    submit_button.click(fn=ask_questions, inputs=question_input, outputs=[answer_output, sources_output])

print("Launching Gradio app...")
demo.launch()
