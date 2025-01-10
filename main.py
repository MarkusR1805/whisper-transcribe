import whisper
import ffmpeg
import warnings
import os
from datetime import timedelta
import time
from tqdm import tqdm  # Fortschrittsbalken für die Transkription
import torch  # Für CUDA-Abfrage

# Zeitpunkt vom Programmstart
start_time = time.time()

# Unterdrücke Warnungen
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Pfad zur Videodatei
video_file = "test.mp4"

# Pfad zur zu erstellenden Audiodatei
audio_file = "audio.wav"

# Pfade für die Ausgabedateien
original_transcription_file = "transkription_deutsch.txt"
translated_transcription_file = "transkription_englisch.txt"
subtitle_file = "unterschrift_englisch.srt"

# Audio aus Video extrahieren (mit ffmpeg-python)
print("Extrahiere Audio aus Video...")
ffmpeg.input(video_file).output(
    audio_file,  # Ausgabe als WAV-Datei
    acodec='pcm_s16le',  # Unkomprimiertes PCM, 16-Bit
    ar=16000,  # Abtastrate 16 kHz (für Whisper empfohlen)
    ac=1  # Mono
).run(overwrite_output=True)  # Überschreibt bestehende Dateien

print(f"Audio wurde erfolgreich extrahiert und gespeichert unter {audio_file}.")

# Überprüfe, ob die Audiodatei existiert
if not os.path.exists(audio_file):
    print(f"Die Audiodatei {audio_file} wurde nicht gefunden.")
    exit(1)

# Gerät (CUDA bevorzugt, MPS ignoriert)
if torch.cuda.is_available():
    device = "cuda"
    print("CUDA verfügbar – Modell wird auf der GPU ausgeführt.")
else:
    device = "cpu"
    print("CUDA nicht verfügbar – Modell wird auf der CPU ausgeführt (MPS wird ignoriert).")

# Whisper-Modell laden und auf gewähltes Gerät verschieben
model = whisper.load_model("medium", device=device)
"""
tiny-Modell (schnell, niedrige Genauigkeit)
model = whisper.load_model("tiny", device=device)

base-Modell (gute Balance zwischen Geschwindigkeit und Genauigkeit)
model = whisper.load_model("base", device="device")

small-Modell (mittlere Genauigkeit, moderate Geschwindigkeit)
model = whisper.load_model("small", device="device")

medium-Modell (hohe Genauigkeit, langsamer als small)
model = whisper.load_model("medium", device="device")

large-Modell (höchste Genauigkeit, langsamste Geschwindigkeit)
model = whisper.load_model("large", device="device")

large-v2-Modell (aktualisierte Version von large)
model = whisper.load_model("large-v2", device="device")

small.en-Modell (speziell für Englisch optimiert)
model = whisper.load_model("small.en", device="device")
"""

# Transkription auf Deutsch (mit Fortschrittsbalken)
print("Starte Transkription...")
result_de = {}
for _ in tqdm(range(1), desc="Transkription läuft"):
    result_de = model.transcribe(audio_file, language="de")

# Übersetzung ins Englische (mit Fortschrittsbalken)
print("Starte Übersetzung...")
result_en = {}
for _ in tqdm(range(1), desc="Übersetzung läuft"):
    result_en = model.transcribe(audio_file, task="translate", language="de")

# Funktion zum Erstellen der SRT-Datei
def generate_srt(transcription, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(transcription['segments'], start=1):
            start_time = str(timedelta(seconds=segment['start'])).replace('.', ',')[:-3]
            end_time = str(timedelta(seconds=segment['end'])).replace('.', ',')[:-3]
            text = segment['text'].strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

# Speichere die Original-Transkription
with open(original_transcription_file, "w", encoding="utf-8") as f:
    f.write(result_de["text"])

print(f"Originaltranskription wurde gespeichert unter {original_transcription_file}.")

# Speichere die Übersetzung
with open(translated_transcription_file, "w", encoding="utf-8") as f:
    f.write(result_en["text"])

print(f"Übersetzung wurde gespeichert unter {translated_transcription_file}.")

# SRT-Datei generieren
generate_srt(result_en, subtitle_file)

print(f"Untertitel wurden gespeichert unter {subtitle_file}.")

end_time = time.time()
duration = end_time - start_time
print(f"Das Transkribieren dauerte {duration:.2f} Sekunden.")
