# Bomba AI - RAG + LoRA LLM Chatbot

[PL] Poniżej znajduje się opis w języku polskim.
[EN] English description is available below the Polish section.

---

## [PL] Opis Projektu

**Status projektu:** W budowie / Work in Progress.
**Uwaga:** Projekt jest w fazie aktywnego rozwoju. Obecna wersja kodu służy do celów demonstracyjnych i przeglądu architektury. Uruchomienie aplikacji nie jest zalecane bez lokalnego wygenerowania niezbędnych zbiorów danych.

### Cel
Projekt Bomba AI to zaawansowane przedsięwzięcie mające na celu stworzenie chatbota opartego na modelu językowym (LLM), który precyzyjnie emuluje styl wypowiedzi specyficznej postaci fikcyjnej, w tym wypadku Kapitana Bomby, jednocześnie zachowując wysoką merytorykę odpowiedzi na temat uniwersum serialu.


### Architektura i Technologie

Projekt wykorzystuje nowoczesny stos technologiczny skupiony wokół lokalnego przetwarzania danych i optymalizacji zasobów (uruchomienie na konsumenckim GPU).

1.  **LLM & Fine-tuning (LoRA/QLoRA):**
    * **Model bazowy:** `speakleash/bielik-7b-instruct-v0.1` (polski model językowy).
    * **Technika:** Low-Rank Adaptation (LoRA) w celu nadania modelowi unikalnego stylu wypowiedzi ("duszy") bez konieczności retrenowania całego modelu.
    * **Narzędzia:** Biblioteka `Unsloth` do optymalizacji zużycia VRAM i przyspieszenia treningu.

2.  **Retrieval-Augmented Generation (RAG):**
    * **Cel:** Dostarczenie modelowi twardych faktów ("mózgu") i eliminacja halucynacji.
    * **Wyszukiwanie:** `LlamaIndex` oraz baza wektorowa `ChromaDB`.
    * **Embeddings:** Wykorzystanie zaawansowanych modeli embeddingów dla języka polskiego (`sdadas/mmlw-retrieval-roberta-large`).

3.  **Data Engineering & ETL Pipeline:**
    * **Pozyskiwanie danych:** Dedykowany proces pobierania i przetwarzania materiałów audio-wideo przy użyciu narzędzia `yt-dlp`.
    * **ASR (Automatic Speech Recognition):** Wykorzystanie modelu `OpenAI Whisper` (uruchamianego lokalnie) do transkrypcji tysięcy linii dialogowych w celu stworzenia wysokiej jakości zbioru treningowego.

4.  **DevOps:**
    * **Konteneryzacja:** Pełna izolacja środowiska przy użyciu `Docker` oraz `NVIDIA Container Toolkit` dla zapewnienia dostępu do GPU wewnątrz kontenera.

### Zastrzeżenie Prawne (Legal Disclaimer)

To repozytorium zawiera wyłącznie kod źródłowy (skrypty Python, konfigurację Docker). **W repozytorium NIE znajdują się żadne zbiory danych, pliki audio, transkrypcje dialogów ani teksty pochodzące z serwisu Fandom.**

Postać Kapitana Bomby, scenariusze oraz dialogi są utworami chronionymi prawem autorskim. Zbiory danych służące do treningu modelu zostały wygenerowane lokalnie wyłącznie w celach edukacyjnych i eksperymentalnych autora i nie są udostępniane publicznie.

### Uruchomienie i Instalacja

Aktualnie repozytorium służy jako portfolio kodu. Sklonowanie i uruchomienie projektu nie przyniesie oczekiwanych rezultatów bez posiadania lokalnie wygenerowanych baz wektorowych oraz adapterów LoRA, które nie są dołączone do repozytorium.

---

## [EN] Project Description

**Project Status:** Work in Progress.
**Note:** This project is under active development. The current codebase is for demonstration and architectural review purposes. Running the application is not recommended without locally generating the necessary datasets.

### Goal
Bomba AI is an advanced engineering project aimed at creating a chatbot based on a Large Language Model (LLM) that precisely emulates the speaking style of a specific fictional character (Kapitan Bomba) while maintaining high factual accuracy regarding the series' universe.

The project addresses the classic LLM hallucination problem through a hybrid approach: separating "style" (fine-tuning) from "knowledge" (RAG).

### Architecture and Technologies

The project utilizes a modern tech stack focused on local data processing and resource optimization (running on consumer-grade GPUs).

1.  **LLM & Fine-tuning (LoRA/QLoRA):**
    * **Base Model:** `speakleash/bielik-7b-instruct-v0.1` (Polish language model).
    * **Technique:** Low-Rank Adaptation (LoRA) to imbue the model with a unique speaking style ("soul") without retraining the entire model.
    * **Tools:** `Unsloth` library for VRAM optimization and training acceleration.

2.  **Retrieval-Augmented Generation (RAG):**
    * **Goal:** Providing the model with hard facts ("brain") and eliminating hallucinations.
    * **Retrieval:** `LlamaIndex` and `ChromaDB` vector database.
    * **Embeddings:** Utilization of advanced embedding models for the Polish language (`sdadas/mmlw-retrieval-roberta-large`).

3.  **Data Engineering & ETL Pipeline:**
    * **Data Ingestion:** Custom audio-video processing pipeline using `yt-dlp`.
    * **ASR (Automatic Speech Recognition):** Utilization of the `OpenAI Whisper` model (running locally) to transcribe thousands of dialogue lines to create a high-quality training dataset.

4.  **DevOps:**
    * **Containerization:** Full environment isolation using `Docker` and `NVIDIA Container Toolkit` to ensure GPU access within the container.

### Legal Disclaimer

This repository contains only source code (Python scripts, Docker configuration). **The repository DOES NOT contain any datasets, audio files, dialogue transcriptions, or texts derived from the Fandom service.**

The character Kapitan Bomba, scripts, and dialogues are copyright-protected works. The datasets used for model training were generated locally solely for the author's educational and experimental purposes and are not shared publicly.

### Installation and Usage

Currently, this repository serves as a code portfolio. Cloning and running the project will not yield the expected results without possessing the locally generated vector databases and LoRA adapters, which are not included in the repository.