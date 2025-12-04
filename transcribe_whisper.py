import os
import sys
import json
from dotenv import load_dotenv
from tqdm import tqdm
from faster_whisper import WhisperModel


# --- FIX DLA WINDOWSA ---
def configure_nvidia_libraries():
    """
    Automatycznie dodaje ścieżki DLL Nvidii na Windowsie.
    """
    try:
        base_prefix = os.path.dirname(os.path.dirname(sys.executable))
        nvidia_path = os.path.join(base_prefix, 'Lib', 'site-packages', 'nvidia')
    except Exception as e:
        return

    if not os.path.exists(nvidia_path):
        return

    target_dlls = {'cudnn_ops64_9.dll', 'cublas64_12.dll'}
    registered_paths = set()

    for root, dirs, files in os.walk(nvidia_path):
        for filename in files:
            if filename in target_dlls:
                if root not in registered_paths:
                    try:
                        os.add_dll_directory(root)
                        registered_paths.add(root)
                        print(f"[WINDOWS FIX] Dodano DLL: {root}")
                    except:
                        pass


# Uruchom naprawę TYLKO jeśli jesteśmy na Windowsie (nt)
if os.name == 'nt':
    configure_nvidia_libraries()
# ------------------------

load_dotenv()

TRANSCRIPTION_OUTPUT_DIR = os.getenv('TRANSCRIPTION_OUTPUT_DIR', 'transcriptions')
AUDIO_SET = os.getenv('AUDIO_SET', 'audio')

# Upewnij się, że folder wyjściowy istnieje
if not os.path.exists(TRANSCRIPTION_OUTPUT_DIR):
    os.makedirs(TRANSCRIPTION_OUTPUT_DIR)

print(f"Szukam plików w: {AUDIO_SET}")
files = [f for f in os.listdir(AUDIO_SET) if f.endswith(".mp3")]
print(f"Znaleziono {len(files)} plików MP3.")

# Sortowanie alfabetyczne jest bezpieczniejsze dla nazw z yt-dlp
files.sort()

model_size = "large-v3"
print(f"Inicjalizacja modelu: {model_size}...")
# compute_type="int8" jest bezpieczniejszy dla 8GB VRAM niż int8_float16
model = WhisperModel(model_size, device="cuda", compute_type="int8")
print("Model gotowy.")

print("Rozpoczynam transkrypcję...")
for filename in tqdm(files):
    input_path = os.path.join(AUDIO_SET, filename)
    output_filename = filename.replace(".mp3", ".json")
    output_path = os.path.join(TRANSCRIPTION_OUTPUT_DIR, output_filename)

    # Jeśli plik już istnieje, pomiń go (przydatne przy restarcie)
    if os.path.exists(output_path):
        continue

    try:
        segments, info = model.transcribe(input_path, beam_size=5, language="pl", vad_filter=True)

        transcript_data = []
        # Pętla generująca tekst
        for segment in segments:
            transcript_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        # Zapis do pliku JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Błąd przy pliku {filename}: {e}")

print("Zakończono!")