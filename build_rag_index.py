import chromadb
import json
import os
import shutil
import sys
import torch
from tqdm import tqdm
from typing import List, Dict, Any

from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- KONFIGURACJA ---
INPUT_DIR = "lore_extracted"
DB_DIRECTORY = "./chroma_db"
EMBED_MODEL_NAME = "sdadas/mmlw-retrieval-roberta-large"


def get_optimal_device() -> str:
    """
    Dynamicznie dobiera urządzenie obliczeniowe.
    Unika hardcodowania 'cuda', co zapobiega błędom na CPU/Mac.
    """
    if torch.cuda.is_available():
        return "cuda"
    # Opcjonalnie: można dodać obsługę 'mps' dla Mac M1/M2/M3
    # if torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def _create_doc(text: str, specific_metadata: Dict[str, Any], base_metadata: Dict[str, Any]) -> Document:
    """
    Helper function (DRY): Tworzy obiekt Document łącząc metadane bazowe i specyficzne.
    """
    # Kopiujemy słownik, aby uniknąć mutacji referencji (częsty błąd w Pythonie)
    meta = base_metadata.copy()
    meta.update(specific_metadata)

    # Zabezpieczenie przed pustym tekstem (na wypadek błędów logicznych)
    if not text or not isinstance(text, str):
        text = "Empty content"

    return Document(text=text, metadata=meta)


def load_documents_from_json(directory: str) -> List[Document]:
    """
    Wczytuje pliki JSON i konwertuje je na semantyczne dokumenty LlamaIndex.
    Zawiera obsługę błędów (try-except) i bezpieczny dostęp do danych (.get).
    """
    llama_documents = []

    # Sprawdzenie czy katalog istnieje
    if not os.path.exists(directory):
        print(f"BŁĄD: Katalog {directory} nie istnieje.")
        return []

    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    print(f"Przetwarzanie {len(files)} plików JSON na semantyczne dokumenty...")

    for filename in tqdm(files):
        path = os.path.join(directory, filename)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Bezpieczne pobieranie metadanych podstawowych
            base_metadata = {
                "episode_id": data.get("episode_id", "Unknown"),
                "title": data.get("title", "Unknown"),
                "source_file": filename
            }

            # 1. SYNOPSIS
            synopsis_text = data.get("synopsis")
            if synopsis_text:
                llama_documents.append(_create_doc(
                    text=synopsis_text,
                    specific_metadata={"type": "synopsis"},
                    base_metadata=base_metadata
                ))

            # 2. LORE FACTS
            for fact in data.get("lore_facts", []):
                # Używamy .get() dla bezpieczeństwa
                cat = fact.get("category", "General")
                content = fact.get("fact", "")

                if content:
                    llama_documents.append(_create_doc(
                        text=f"Fakt ({cat}): {content}",
                        specific_metadata={"type": "lore_fact", "category": cat},
                        base_metadata=base_metadata
                    ))

            # 3. CHARACTER ACTIONS
            for char in data.get("character_actions", []):
                name = char.get("name", "Unknown")
                role = char.get("role_in_episode", "")
                traits = ", ".join(char.get("traits_exhibited", []))

                text_content = f"Postać: {name}. Rola: {role} Cechy: {traits}."

                llama_documents.append(_create_doc(
                    text=text_content,
                    specific_metadata={"type": "character_profile", "character": name},
                    base_metadata=base_metadata
                ))

            # 4. QUOTES
            quotes_data = data.get("quotes", {})

            # Attributed Quotes
            for quote in quotes_data.get("attributed_quotes", []):
                speaker = quote.get("speaker", "Unknown")
                text = quote.get("text", "")
                context = quote.get("context", "")
                confidence = quote.get("confidence", "Medium")

                text_content = f"{speaker} powiedział: \"{text}\""
                if context:
                    text_content += f" (Kontekst: {context})"

                llama_documents.append(_create_doc(
                    text=text_content,
                    specific_metadata={
                        "type": "quote",
                        "speaker": speaker,
                        "confidence": confidence
                    },
                    base_metadata=base_metadata
                ))

            # Unattributed Gems
            for gem in quotes_data.get("unattributed_gems", []):
                if gem:  # Ignoruj puste stringi
                    llama_documents.append(_create_doc(
                        text=f"Cytat z uniwersum: \"{gem}\"",
                        specific_metadata={"type": "quote_unattributed"},
                        base_metadata=base_metadata
                    ))

        except json.JSONDecodeError:
            print(f"\nBŁĄD: Plik {filename} jest uszkodzonym JSON-em. Pomijam.", file=sys.stderr)
            continue
        except Exception as e:
            print(f"\nBŁĄD: Nieoczekiwany problem z plikiem {filename}: {e}", file=sys.stderr)
            continue

    return llama_documents


def build_index():
    # 0. Safety Clean
    if os.path.exists(DB_DIRECTORY):
        print(f"Czyszczenie starego indeksu w '{DB_DIRECTORY}'...")
        try:
            shutil.rmtree(DB_DIRECTORY)
        except OSError as e:
            print(f"Nie udało się usunąć folderu: {e}")
            # Czasami Windows blokuje pliki, jeśli proces chroma wciąż działa w tle
            # W Dockerze/WSL powinno być ok.

    if not os.path.exists(INPUT_DIR):
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono folderu {INPUT_DIR}.")
        sys.exit(1)

    # 1. Device Selection
    device = get_optimal_device()
    print(f"Wybrane urządzenie obliczeniowe: {device.upper()}")

    # 2. Load Embeddings
    print(f"Ładowanie modelu embeddingów: {EMBED_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device=device
    )

    # 3. Prepare ChromaDB
    print("Inicjalizacja ChromaDB...")
    db = chromadb.PersistentClient(path=DB_DIRECTORY)
    chroma_collection = db.get_or_create_collection("bomba_lore")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Load & Process Data
    documents = load_documents_from_json(INPUT_DIR)

    if not documents:
        print("Nie znaleziono żadnych poprawnych dokumentów do zaindeksowania.")
        return

    print(f"Wygenerowano {len(documents)} semantycznych fragmentów (chunks).")

    # 5. Indexing
    print("Budowanie indeksu wektorowego...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print("\n--- SUKCES ---")
    print(f"Baza wiedzy została zapisana w: {DB_DIRECTORY}")


if __name__ == "__main__":
    build_index()