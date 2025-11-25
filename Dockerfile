# Krok 1: Wybieramy Obraz Bazowy
# To jest nasze fundament - system operacyjny (Linux) z zainstalowanym CUDA, Pythonem 3.10,
# PyTorchem, i co najważniejsze, biblioteką Unsloth i jej zależnościami.
FROM unsloth/unsloth:latest

# Krok 2: Ustawiamy Katalog Roboczy
# Tworzymy folder /app wewnątrz kontenera i mówimy Dockerowi, 
# że to będzie nasz domyślny folder do pracy.
WORKDIR /app

# Krok 3: Kopiujemy i Instalujemy Zależności
# Najpierw kopiujemy tylko plik requirements.txt. 
# Robimy to jako osobny krok, aby Docker mógł "zcache'ować" (zapamiętać)
# zainstalowane pakiety i nie instalował ich za każdym razem, gdy zmienimy skrypt .py.
COPY requirements.txt .

# Uruchamiamy instalację pip dla pakietów z pliku requirements.txt.
# --no-cache-dir oszczędza miejsce.
RUN pip install -r requirements.txt --no-cache-dir

# Krok 4: Ustawiamy Zmienne Środowiskowe (Dobra Praktyka)
# Mówimy bibliotekom, gdzie mają trzymać pobrane modele.
# Ułatwi to zarządzanie cache'em, jeśli będziemy chcieli to później zoptymalizować.
ENV HF_HOME=/app/huggingface-cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/sentence-transformers-cache

# Krok 5: Kopiujemy Resztę Projektu
# Kopiujemy wszystko inne z naszego folderu (skrypty .py, folder /data)
# do katalogu roboczego /app wewnątrz kontenera.
COPY . .

# Krok 6: Domyślna Komenda
# Mówimy Dockerowi, co ma uruchomić, gdy powiemy mu "docker run ...".
# W naszym przypadku uruchomi interaktywny terminal "bash",
# abyśmy mogli ręcznie wpisywać polecenia python.
CMD [ "bash" ]