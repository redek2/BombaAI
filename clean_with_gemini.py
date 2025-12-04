import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- KONFIGURACJA ---
INPUT_DIR = "transcriptions"
OUTPUT_DIR = "transcriptions_clean"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("BŁĄD: Nie znaleziono zmiennej środowiskowej GEMINI_API_KEY.")
    exit(1)

# Konfiguracja klienta
genai.configure(api_key=api_key)

# Ustawienia bezpieczeństwa - WYŁĄCZAMY BLOKADY
# To jest kluczowe dla Kapitana Bomby. Bez tego model odrzuci 90% tekstów.
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Inicjalizacja modelu
model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)

# Prompt Systemowy - Instrukcja dla "Korektora"
SYSTEM_PROMPT = """
Jesteś profesjonalnym korektorem transkrypcji ASR (Automatic Speech Recognition) dla serialu "Kapitan Bomba".
Twoim zadaniem jest poprawienie błędów w dostarczonym tekście JSON, zachowując 100% wulgarności i stylu oryginału.

ZASADY:
1. Popraw literówki i błędy fonetyczne (np. "dziadnięty" -> "dziabnięty", "Udasha" -> "Janusza", "koszmitów" -> "kosmitów").
2. Popraw nazwy własne (np. "Sraturn", "Kurvinox", "RKS Huwdu", "Torpeda", "Kutanoid", "Skurwol").
3. USUŃ CENZURĘ: Jeśli widzisz "k***a", "j*****e", zamień na pełne wulgaryzmy ("kurwa", "jebanie").
4. NIE ZMIENIAJ struktury JSONA. Zwróć dokładnie taką samą listę obiektów, poprawiając tylko pole "text".
5. Nie dopisuj nic od siebie. Nie bądź kreatywny. Bądź precyzyjny.

Oto surowy JSON z błędami:
"""


def clean_file_with_gemini(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    # 1. Wczytaj surowy plik
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Optymalizacja: Wysyłamy cały plik jako jeden prompt (Flash ma duże okno kontekstowe)
    # Konwertujemy JSON do stringa, żeby wysłać go jako tekst
    json_string = json.dumps(raw_data, ensure_ascii=False)

    # 2. Wyślij do API
    try:
        response = model.generate_content(SYSTEM_PROMPT + json_string)

        # 3. Oczyść odpowiedź (czasami model dodaje ```json ... ```)
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        # 4. Parsuj z powrotem do obiektu
        corrected_data = json.loads(cleaned_text)

        # 5. Zapisz
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(corrected_data, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f"\nBłąd przy pliku {filename}: {e}")
        # Jeśli model odrzucił treść (Safety), spróbujmy zapisać to co mamy, żeby nie stracić
        return False


# --- GŁÓWNA PĘTLA ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    print(f"--- Rozpoczynam czyszczenie {len(files)} plików przy użyciu Gemini API ---")

    for filename in tqdm(files):
        # Sprawdź, czy już nie zrobione (oszczędność API)
        if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
            print(filename, "istnieje.")
            continue

        success = clean_file_with_gemini(filename)

        if success:
            # Ważne: Rate Limiting dla Free Tier
            time.sleep(7)
        else:
            print(f"Pominięto plik {filename} z powodu błędu.")

    print("\nZakończono proces czyszczenia.")