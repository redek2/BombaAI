import os
import json
import time
import re
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


# --- SCHEMA DEFINITION (Bez zmian) ---

class CharacterAction(BaseModel):
    name: str = Field(description="Imię postaci (np. Kapitan Bomba, Torpeda, Janusz).")
    role_in_episode: str = Field(description="Krótki opis co postać robiła w odcinku.")
    traits_exhibited: list[str] = Field(description="Cechy charakteru ujawnione w tym odcinku (np. chciwość, agresja).")


class LoreFact(BaseModel):
    category: str = Field(description="Kategoria faktu: Technology, Species, Location, History.")
    fact: str = Field(description="Konkretny fakt o świecie (np. 'Kurvinoxy mają kieszonkę za rozporkiem').")


class AttributedQuote(BaseModel):
    speaker: str = Field(description="Kto mówi (tylko jeśli pewność jest wysoka).")
    text: str = Field(description="Treść cytatu.")
    confidence: Literal["High", "Medium"] = Field(description="Poziom pewności przypisania (High/Medium).")
    context: str | None = Field(description="Kontekst sytuacyjny (do kogo, w jakiej sytuacji).")


class QuotesAnalysis(BaseModel):
    episode_vocabulary: list[str] = Field(description="Lista charakterystycznych słów/wulgaryzmów dla tego odcinka.")
    attributed_quotes: list[AttributedQuote] = Field(description="Cytaty przypisane do konkretnych postaci.")
    unattributed_gems: list[str] = Field(description="Kultowe cytaty, których autora nie da się jednoznacznie ustalić.")


class EpisodeAnalysis(BaseModel):
    episode_id: str = Field(description="Numer lub identyfikator odcinka, jeśli dostępny.")
    title: str = Field(description="Tytuł odcinka.")
    synopsis: str = Field(description="Streszczenie fabuły (2-3 zdania).")
    character_actions: list[CharacterAction]
    lore_facts: list[LoreFact]
    quotes: QuotesAnalysis


# --- CLIENT CONFIG ---

INPUT_DIR = "transcriptions_clean"
OUTPUT_DIR = "lore_extracted"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No API key set.")
    exit(1)

client = genai.Client(api_key=api_key)

safety_settings = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

SYSTEM_PROMPT = """
Jesteś Głównym Archiwistą Uniwersum "Kapitan Bomba". Twoim zadaniem jest przekształcenie surowej transkrypcji audio (JSON) w ustrukturyzowaną bazę wiedzy (Encyklopedię).

Twoim priorytetem jest wierność materiałowi źródłowemu, który jest wulgarny, absuradny i pełen czarnego humoru. Nie cenzuruj niczego. Zachowaj oryginalną pisownię nazw własnych (np. Kurvinox, Skurwol, RKS Huwdu).

ZASADY ANALIZY (Ściśle przestrzegaj):
1. **LORE & FACTS (Fakty):** Wyciągaj twarde fakty o świecie (technologia, biologia kosmitów). Ignoruj bełkot bez znaczenia.
2. **KONSERWATYWNA ATRYBUCJA (Kto to powiedział?):** - Transkrypcja nie ma podziału na role. Musisz wydedukować mówcę z kontekstu.
   - ZASADA "HIGH PRECISION": Przypisz cytat w `attributed_quotes` TYLKO WTEDY, gdy jesteś w 100% pewien (pada imię, jasny kontekst).
   - Wątpliwości? Wrzuć do `unattributed_gems`. Nie zgaduj!
3. **STYL I SŁOWNICTWO:** Wyłapuj unikalne słownictwo ("tępy chuj", "napierdalać", "Kurwinox).

Otrzymasz plik JSON z segmentami transkrypcji. Przeanalizuj całość.
"""


# --- PROCESSING FUNCTION ---

def process_file(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    # 1. Ekstrakcja metadanych (Regex Robustness)
    episode_id_match = re.search(r"\(ODC\.\s*(\d+)\)", filename)
    real_episode_id = episode_id_match.group(1) if episode_id_match else "Unknown"

    # Wyciągamy Tytuł (wszystko między myślnikiem a ODC)
    # Obsługuje formaty: "BOMBA - TYTUŁ (ODC...", "BOMBA - TYTUŁ | (ODC..."
    real_title = filename.replace(".json", "")  # Default
    try:
        # Usuwamy prefiks jeśli jest
        clean_name = filename.replace("KAPITAN BOMBA - ", "")
        # Dzielimy po znaku otwarcia nawiasu ID lub pionowej kreski
        # Regex: Znajdź ' (' lub ' ｜' lub ' |' przed 'ODC'
        split_match = re.split(r"(\s*[\|｜]\s*)?\(ODC", clean_name)
        if split_match:
            real_title = split_match[0].strip()
    except Exception:
        pass

    # 2. Wczytanie danych
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_transcription_text = json.dumps(data, ensure_ascii=False)

    # 3. Call API
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[SYSTEM_PROMPT, raw_transcription_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EpisodeAnalysis,
                safety_settings=safety_settings,
                temperature=0.2
            )
        )

        if response.parsed:
            final_data = response.parsed
            final_data.episode_id = real_episode_id
            final_data.title = real_title

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_data.model_dump_json(indent=2))
            return True
        else:
            print(f"Błąd: Model zwrócił pustą odpowiedź dla {filename}")
            return False

    except Exception as e:
        print(f"API Error dla {filename}: {e}")
        return False


# --- MAIN PRODUCTION LOOP ---

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Pobieramy listę wszystkich plików
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    # Sortujemy, żeby robić po kolei (opcjonalne, ale ładniej wygląda)
    files.sort()

    print(f"--- ROZPOCZYNAM BUDOWĘ ENCYKLOPEDII ---")
    print(f"Znaleziono plików: {len(files)}")
    print(f"Model: Gemini 2.5 Flash | Output: {OUTPUT_DIR}/")

    success_count = 0
    fail_count = 0

    # Pasek postępu
    pbar = tqdm(files)
    for filename in pbar:
        pbar.set_description(f"Przetwarzanie: {filename[:30]}...")

        # Idempotentność: Jeśli plik istnieje, pomiń
        if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
            continue

        success = process_file(filename)

        if success:
            success_count += 1
            # Rate Limiting: 4 sekundy przerwy między żądaniami (Free Tier ~15 RPM)
            # Jeśli masz płatne API, możesz zmniejszyć do 0.5s lub 1s
            time.sleep(4)
        else:
            fail_count += 1
            # Zapiszmy błędy do logu, żeby wiedzieć co powtórzyć
            with open("failed_files.txt", "a") as log:
                log.write(f"{filename}\n")

    print(f"\n--- ZAKOŃCZONO ---")
    print(f"Sukcesy: {success_count}")
    print(f"Błędy: {fail_count}")
    print(f"Sprawdź folder: {OUTPUT_DIR}")