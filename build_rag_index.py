import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from transformers import AutoTokenizer
import sys
import os

# --- Configuration ---
INPUT_FILE = "data/RAG_FANDOM_DATA_CLEAN.txt"
DB_DIRECTORY = "./chroma_db"  # Directory to save the persistent database
EMBED_MODEL_NAME = "sdadas/mmlw-retrieval-roberta-large"
CHUNK_SIZE = 512  # The size of each text "chunk" in tokens
CHUNK_OVERLAP = 32  # How much overlap between chunks


def build_persistent_index():
    """
    Builds a persistent vector index from our data file and saves it to disk.
    """
    print(f"--- Starting Index Build ---")

    # Check if the input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}", file=sys.stderr)
        print("Please make sure the file exists in the same directory.", file=sys.stderr)
        sys.exit(1)

    # --- Step 1: Load the Embedding Model & Tokenizer ---
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device="cpu"
    )
    print("Embedding model loaded.")

    print(f"Loading tokenizer for: {EMBED_MODEL_NAME}")
    # Wczytujemy tokenizer pasujący do naszego modelu embeddingów
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    def hf_tokenizer_fn(text: str):
        # return list of token ids (TokenTextSplitter only needs lenght)
        return tokenizer.encode(text, add_special_tokens=False)

    print("Tokenizer loaded.")

    # --- Step 2: Set up the Vector Database (ChromaDB) ---
    # (Ten krok jest taki sam, ale jest w nowym bloku dla porządku)
    print(f"Initializing persistent vector database at: {DB_DIRECTORY}")
    db = chromadb.PersistentClient(path=DB_DIRECTORY)
    chroma_collection = db.get_or_create_collection("bomba_lore")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # --- Step 3: Load the Data ---
    # (Ten krok jest taki sam)
    print(f"Loading data from: {INPUT_FILE}")
    documents = SimpleDirectoryReader(
        input_files=[INPUT_FILE]
    ).load_data()
    print(f"Data loaded. Found {len(documents)} document(s).")

    # --- Step 4: Define the Indexing Pipeline ---
    # This ties everything together.

    # 4a. Node Parser (Chunking)
    # Używamy SentenceSplitter i przekazujemy mu DOKŁADNY tokenizer
    # To gwarantuje, że "chunk_size=256" będzie mierzone poprawną miarką.
    print("Initializing node parser (SentenceSplitter) with tokenizer.")
    node_parser = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=hf_tokenizer_fn
    )

    # 4b. Storage Context
    # Tells LlamaIndex WHERE to store the data (in our ChromaDB).
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # --- Step 5: Build the Index ---
    # This is the main event!
    # LlamaIndex will automatically:
    # 1. Take `documents`
    # 2. Split them into chunks (using `node_parser`)
    # 3. Convert each chunk to an embedding (using `embed_model`)
    # 4. Store the embedding + chunk text in our `vector_store` (ChromaDB)
    print("Building index... This may take a few minutes...")
    print(f"Chunk Size: {CHUNK_SIZE}, Chunk Overlap: {CHUNK_OVERLAP}")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        node_parser=node_parser,
        show_progress=True
    )

    print("\n--- Index Build Complete! ---")
    print(f"Vector database is persistently stored in '{DB_DIRECTORY}'")
    #print(f"Total nodes indexed: {len(index.index_struct.nodes)}")


# This block runs only when you execute the script directly
if __name__ == "__main__":
    build_persistent_index()