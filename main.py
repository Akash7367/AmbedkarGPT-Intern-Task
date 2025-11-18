import os
import argparse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


# Config
SPEECH_FILE = "speech.txt"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore(speech_path: str, persist_directory: str) -> Chroma:
    """Load the speech, split, embed, and persist the Chroma vectorstore.
    If the persisted DB already exists, this function will load it instead of rebuilding.
    """
    # If already persisted, load and return
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing Chroma DB from {persist_directory}...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectordb

    print("Creating new Chroma DB and indexing documents...")

    # 1) Load
    loader = TextLoader(speech_path, encoding="utf-8")
    docs = loader.load()

    # 2) Split into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    docs = splitter.split_documents(docs)

    # 3) Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4) Create ChromaDB
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    print(f"Persisted Chroma DB to {persist_directory}")
    return vectordb


def build_qa_chain(vectordb: Chroma) -> RetrievalQA:
    """Create Ollama LLM and RetrievalQA chain using the vectordb's retriever."""
    # Create Ollama LLM wrapper. Assumes Ollama is installed and `mistral` model is pulled locally.
    llm = Ollama(model="mistral")

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


def interactive_loop(qa_chain: RetrievalQA):
    print("\nAmbedkarGPT — ask questions about the provided speech. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye.")
            break

        # Run retrieval + generation
        result = qa_chain(query)
        answer = result.get("result") or result.get("answer") or "(no answer returned)"
        print("\nAnswer:\n", answer)

        # Optionally show sources (retrieved chunks)
        docs = result.get("source_documents") or []
        if docs:
            print("\n-- Retrieved context (for transparency) --")
            for i, d in enumerate(docs, 1):
                print(f"\n[{i}] (source length={len(d.page_content)} chars)\n{d.page_content}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AmbedkarGPT - simple RAG CLI")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vectorstore (re-embed)")
    parser.add_argument("--speech", type=str, default=SPEECH_FILE, help="Path to speech.txt")
    parser.add_argument("--chroma-dir", type=str, default=CHROMA_DIR, help="Chroma persist directory")
    args = parser.parse_args()

    if not os.path.exists(args.speech):
        raise FileNotFoundError(f"Speech file not found: {args.speech}. Please create speech.txt in the repo root.")

    if args.rebuild and os.path.exists(args.chroma_dir):
        # simple way to force rebuild
        import shutil
        print("Removing existing Chroma DB to force rebuild...")
        shutil.rmtree(args.chroma_dir)

    vectordb = build_vectorstore(args.speech, args.chroma_dir)
    qa_chain = build_qa_chain(vectordb)
    interactive_loop(qa_chain)