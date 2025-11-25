import torch
from unsloth import FastLanguageModel
import chromadb
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
import sys

# --- 1. Konfiguracja ---
MODEL_DO_ZALADOWANIA = "./lora_adapter"
DB_DIRECTORY = "./chroma_db"
EMBED_MODEL_NAME = "sdadas/mmlw-retrieval-roberta-large"
SIMILARITY_TOP_K = 3

print("--- 🚀 Startowanie Bota Bomby (Tryb Debugowania) ---")
print("To może potrwać kilka minut, model musi załadować się do VRAM.")

# --- 2. Ładowanie modelu LoRA (na GPU) ---
print(f"Ładowanie modelu i adaptera z: {MODEL_DO_ZALADOWANIA}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_DO_ZALADOWANIA,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
print("✅ Model i adapter LoRA załadowane na GPU.")

# --- 3. Ładowanie bazy RAG (na CPU) ---
print(f"Ładowanie bazy wektorowej RAG z: {DB_DIRECTORY}")
db = chromadb.PersistentClient(path=DB_DIRECTORY)
chroma_collection = db.get_collection("bomba_lore")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

print(f"Ładowanie modelu embeddingów (na CPU): {EMBED_MODEL_NAME}")
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device="cpu")

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

print(f"Inicjalizacja retrievera RAG z top_k = {SIMILARITY_TOP_K}")
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=SIMILARITY_TOP_K,
    embed_model=embed_model,
)
print("✅ Baza RAG gotowa.")

# --- 4. Formatka Promptu ---
prompt_template = """<s>### Instrukcja:
Na podstawie poniższego kontekstu, odpowiedz na pytanie.
Użyj wulgarnego i bezpośredniego stylu Kapitana Bomby.

Kontekst:
{kontekst}

Pytanie:
{pytanie}

### Odpowiedź:
"""

# --- 5. Pętla Czat-bota ---
print("\n--- ✅ Bot gotowy. Zadaj pytanie. Wpisz 'wyjscie' aby zakończyć. ---")

while True:
    try:
        pytanie_uzytkownika = input("\nTy: ")
        if pytanie_uzytkownika.lower() in ["wyjscie", "exit", "quit", "koniec"]:
            print("--- 🛑 Zamykanie bota. ---")
            break

        # Krok A: Znajdź kontekst w bazie RAG
        print("...myślę (szukam w bazie RAG)...")
        wyniki_retrievera = retriever.retrieve(pytanie_uzytkownika)
        retrieved_texts = [wynik.get_text() for wynik in wyniki_retrievera]

        max_model_tokens = 2048
        reserved_for_prompt_and_gen = 512
        allowed_context_tokens = max_model_tokens - reserved_for_prompt_and_gen

        kontekst_rag = ""
        current_tokens = 0
        for txt in retrieved_texts:
            token_ids = tokenizer.encode(txt, add_special_tokens=False)
            tlen = len(token_ids)
            if current_tokens + tlen <= allowed_context_tokens:
                kontekst_rag += txt + "\n\n"
                current_tokens += tlen
            else:
                remaining = allowed_context_tokens - current_tokens
                if remaining > 20:
                    toks = token_ids[:remaining]
                    part_text = tokenizer.decode(toks, skip_special_tokens=True)
                    kontekst_rag += part_text + "\n\n"
                break


        # Krok B: Zbuduj pełny prompt
        finalny_prompt = prompt_template.format(
            kontekst=kontekst_rag,
            pytanie=pytanie_uzytkownika
        )

        # Krok C: Wygeneruj odpowiedź za pomocą modelu LoRA
        print("...myślę (generuję odpowiedź LoRA)...")

        # Teraz tokenizujemy finalny prompt - mamy gwarancję, że jest < 2048
        inputs = tokenizer([finalny_prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        # Krok D: Zdekoduj i wydrukuj odpowiedź
        odpowiedz = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            tylko_odpowiedz = odpowiedz.split("### Odpowiedź:")[1].strip()
            print(f"\nBomba: {tylko_odpowiedz}")
        except IndexError:
            print(f"\nBomba (błąd dekodowania): {odpowiedz}")

    except KeyboardInterrupt:
        print("\n--- 🛑 Przerywanie. Wpisz 'wyjscie' aby zakończyć. ---")
    except Exception as e:
        print(f"Wystąpił błąd: {e}", file=sys.stderr)

print("Do widzenia, tępy chuju.")
