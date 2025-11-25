import chromadb
from transformers import AutoTokenizer
import sys

# --- Konfiguracja (musi być taka sama jak w skryptach) ---
DB_PATH = "./chroma_db"
TOKENIZER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def analyze_database():
    print("--- 🕵️‍♂️ Rozpoczynam Analizę Bazy Danych ChromaDB ---")

    try:
        # Używamy tej samej "linijki" (tokenizera) co model RAG
        print(f"Ładuję tokenizer: {TOKENIZER_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print("Tokenizer załadowany.")

        # Łączymy się z *istniejącą* bazą danych
        print(f"Ładuję bazę danych z: {DB_PATH}...")
        db = chromadb.PersistentClient(path=DB_PATH)
        collection = db.get_collection("bomba_lore")
        print("Baza danych załadowana.")

        # Pobieramy WSZYSTKIE dokumenty z bazy
        print("Pobieram *wszystkie* fragmenty z bazy (to może chwilę potrwać)...")
        results = collection.get(include=["documents"])
        documents = results['documents']
        total_chunks = len(documents)

        if total_chunks == 0:
            print("BŁĄD: Baza danych jest pusta!")
            return

        print(f"Pobrano {total_chunks} fragmentów. Analizuję długość każdego z nich...")

        # Mierzymy długość każdego fragmentu
        lengths = []
        for doc in documents:
            # Używamy .encode(), a nie .tokenize(), aby dostać listę ID tokenów
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            lengths.append(len(tokens))

        # Analiza statystyczna
        max_len = max(lengths)
        min_len = min(lengths)
        avg_len = sum(lengths) / total_chunks

        # Szukamy "potwornych fragmentów"
        over_512 = sum(1 for l in lengths if l > 512)
        over_1024 = sum(1 for l in lengths if l > 1024)
        over_2048 = sum(1 for l in lengths if l > 2048) # To są te, które psują nam czat

        print("\n" + "="*50)
        print("--- WYNIKI ANALIZY BAZY DANYCH (chroma_db) ---")
        print(f"Ilość wszystkich fragmentów (chunków): {total_chunks}")
        print(f"Średnia długość fragmentu: {avg_len:.2f} tokenów")
        print(f"Minimalna długość fragmentu: {min_len} tokenów")
        print(f"!!! MAKSYMALNA długość fragmentu: {max_len} tokenów !!!")
        print("-" * 50)
        print(f"Fragmenty dłuższe niż 512 tokenów: {over_512}")
        print(f"Fragmenty dłuższe niż 1024 tokeny: {over_1024}")
        print(f"Fragmenty dłuższe niż 2048 tokenów: {over_2048}")
        print("="*50)

    except Exception as e:
        print(f"Wystąpił błąd podczas analizy: {e}", file=sys.stderr)

if __name__ == "__main__":
    analyze_database()