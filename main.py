import sys
import os
import whisper
import ffmpeg
from datetime import timedelta, datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox, QProgressBar,
    QSlider, QHBoxLayout, QSplitter, QFormLayout, QComboBox, QRadioButton, QButtonGroup, QFrame, QCheckBox
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QRect, QPoint
from PyQt6.QtGui import QScreen
import warnings
import torch
import multiprocessing
import psutil

# Unterdrücke Warnungen
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Verfügbare Whisper-Modelle mit Speicheranforderungen (in MB)
WHISPER_MODELS = {
    "Tiny (schnell, niedrige Genauigkeit)": {"name": "tiny", "memory": 72},
    "Base (ausgewogen)": {"name": "base", "memory": 139},
    "Small (gut)": {"name": "small", "memory": 461},
    "Medium (sehr gut)": {"name": "medium", "memory": 1420},
    "Turbo (sehr schnell, eventuell ungenau)": {"name": "turbo", "memory": 1510},
    "Large (beste Qualität)": {"name": "large", "memory": 2880},
    "Large V2 (beste Qualität, neu)": {"name": "large-v2", "memory": 2870},
    "Large V3 (beste Qualität, neueste)": {"name": "large-v3", "memory": 2880}
}

def check_cuda_availability():
    """Überprüft die Verfügbarkeit von CUDA und gibt Details zurück"""
    if torch.cuda.is_available():
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0) / 1024**2,  # In MB
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**2  # In MB
        }
    return {"available": False}

class DeviceManager:
    """Verwaltet die Geräteauswahl und -konfiguration"""
    def __init__(self):
        self.cuda_info = check_cuda_availability()
        self.current_device = "cuda" if self.cuda_info["available"] else "cpu"
        self.cpu_cores = multiprocessing.cpu_count()

    def get_device_info(self):
        """Gibt Informationen über das aktuelle Gerät zurück"""
        if self.current_device == "cuda":
            return f"GPU: {self.cuda_info['device_name']}"
        return f"CPU: {self.cpu_cores} Kerne verfügbar"

    def set_device(self, device):
        """Setzt das zu verwendende Gerät"""
        if device == "cuda" and not self.cuda_info["available"]:
            raise ValueError("CUDA ist nicht verfügbar")
        self.current_device = device

class TranscriptionWorker(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    time_estimate_signal = pyqtSignal(str)

    def __init__(self, video_path, base_filename, model_name, device_type, keep_wav=False, num_cores_to_use=None):
        super().__init__()
        self.video_path = video_path
        self.base_filename = base_filename
        self.model_name = model_name
        self.device_type = device_type
        self.num_cores_to_use = num_cores_to_use
        self.keep_wav = keep_wav
        self.is_cancelled = False
        self.start_time = None
        self.model = None

    def cancel(self):
        self.is_cancelled = True
        self.status_signal.emit("Transkription wird abgebrochen...")

    def estimate_remaining_time(self, progress):
        if self.start_time and progress > 0:
            elapsed_time = datetime.now() - self.start_time
            estimated_total_time = elapsed_time / (progress / 100)
            remaining_time = estimated_total_time - elapsed_time
            return str(remaining_time).split('.')[0]
        return "Wird berechnet..."

    def run(self):
        try:
            self.start_time = datetime.now()
            self.status_signal.emit("Initialisiere...")

            # Dateipfade erstellen
            output_dir = os.path.dirname(self.base_filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            audio_file = f"{self.base_filename}.wav"
            transcription_file_de = f"{self.base_filename}_de.txt"
            transcription_file_en = f"{self.base_filename}_en.txt"
            subtitle_file = f"{self.base_filename}.srt"

            # Audio extrahieren
            self.status_signal.emit("Extrahiere Audio...")
            self.progress_signal.emit(10)

            if self.is_cancelled:
                raise InterruptedError("Transkription wurde abgebrochen")

            ffmpeg.input(self.video_path).output(
                audio_file, acodec="pcm_s16le", ar=16000, ac=1
            ).run(overwrite_output=True, quiet=True)

            # Gerätekonfiguration
            if self.device_type == "cpu" and self.num_cores_to_use is not None:
                torch.set_num_threads(self.num_cores_to_use)
                self.status_signal.emit(f"Verwende {self.num_cores_to_use} CPU-Kerne")

            # Modell laden
            self.status_signal.emit(f"Lade Whisper-Modell ({self.model_name})...")
            self.progress_signal.emit(30)

            if self.is_cancelled:
                raise InterruptedError("Transkription wurde abgebrochen")

            self.model = whisper.load_model(
                WHISPER_MODELS[self.model_name]["name"],
                device=self.device_type
            )

            # Transkription auf Deutsch
            self.status_signal.emit("Erstelle deutsche Transkription...")
            self.progress_signal.emit(50)

            if self.is_cancelled:
                raise InterruptedError("Transkription wurde abgebrochen")

            result_de = self.model.transcribe(audio_file, language="de")

            # Übersetzung ins Englische
            self.status_signal.emit("Erstelle englische Übersetzung...")
            self.progress_signal.emit(70)

            if self.is_cancelled:
                raise InterruptedError("Transkription wurde abgebrochen")

            # result_en = self.model.transcribe(audio_file, task="translate", language="de")
            result_en = self.model.transcribe(audio_file, task="translate", language="en")

            # Dateien speichern
            self.status_signal.emit("Speichere Dateien...")

            if self.is_cancelled:
                raise InterruptedError("Transkription wurde abgebrochen")

            # Transkriptionen speichern
            with open(transcription_file_de, "w", encoding="utf-8") as f:
                f.write(result_de["text"])
            with open(transcription_file_en, "w", encoding="utf-8") as f:
                f.write(result_en["text"])

            # SRT-Datei generieren
            self.generate_srt(result_en, subtitle_file)

            # Temporäre WAV-Datei löschen
            # Temporäre WAV-Datei behandeln
            if not self.keep_wav and os.path.exists(audio_file):
                os.remove(audio_file)
            elif self.keep_wav:
                self.status_signal.emit(f"WAV-Datei wurde behalten: {audio_file}")

            self.progress_signal.emit(100)
            self.finished_signal.emit(
                f"Transkription abgeschlossen!\nDateien gespeichert als:\n"
                f"- {transcription_file_de}\n"
                f"- {transcription_file_en}\n"
                f"- {subtitle_file}"
            )

        except InterruptedError as e:
            self.error_signal.emit(str(e))
        except Exception as e:
            self.error_signal.emit(f"Fehler: {str(e)}")

    def generate_srt(self, transcription, output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(transcription["segments"], start=1):
                if self.is_cancelled:
                    break
                start_time = timedelta(seconds=segment["start"])
                end_time = timedelta(seconds=segment["end"])
                start_time_str = f"{start_time.seconds//3600:02d}:{(start_time.seconds//60)%60:02d}:{start_time.seconds%60:02d},{start_time.microseconds // 1000:03d}"
                end_time_str = f"{end_time.seconds//3600:02d}:{(end_time.seconds//60)%60:02d}:{end_time.seconds%60:02d},{end_time.microseconds // 1000:03d}"
                text = segment["text"].strip()
                f.write(f"{i}\n{start_time_str} --> {end_time_str}\n{text}\n\n")

class TranscriptionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.device_manager = DeviceManager()
        self.init_ui()
        self.center_window()

    def center_window(self):
        # Zentriert das Fenster auf dem Bildschirm
        screen = QApplication.primaryScreen().geometry()
        window_size = self.geometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
        self.move(x, y)

    def init_ui(self):
        self.setWindowTitle("Video-Transkription mit Whisper")
        self.setFixedSize(575, 965)  # Feste Fenstergröße
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Video-Bereich
        video_container = QFrame()
        video_container.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        video_layout = QVBoxLayout(video_container)

        self.video_widget = QVideoWidget()
        self.video_widget.setFixedHeight(300)
        width = int(300 * (16 / 9))
        self.video_widget.setFixedWidth(width)
        video_layout.addWidget(self.video_widget)
        main_layout.addWidget(video_container)

        # Steuerelemente-Container
        controls_container = QFrame()
        controls_container.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        controls_layout = QVBoxLayout(controls_container)

        # Video-Steuerung
        playback_layout = QHBoxLayout()
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        playback_layout.addWidget(self.btn_play_pause)

        # Lautstärkeregler
        self.label_volume = QLabel("Lautstärke:")
        playback_layout.addWidget(self.label_volume)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.set_volume)
        playback_layout.addWidget(self.volume_slider)
        controls_layout.addLayout(playback_layout)

        # Geräteauswahl
        device_group = QFrame()
        device_layout = QVBoxLayout(device_group)
        device_layout.setSpacing(5)

        device_label = QLabel("Verarbeitungsgerät:")
        device_layout.addWidget(device_label)

        self.device_buttons = QButtonGroup()

        # CPU Radio Button
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setChecked(not self.device_manager.cuda_info["available"])
        self.device_buttons.addButton(self.cpu_radio)
        device_layout.addWidget(self.cpu_radio)

        # GPU Radio Button
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        self.gpu_radio.setEnabled(self.device_manager.cuda_info["available"])
        self.gpu_radio.setChecked(self.device_manager.cuda_info["available"])
        self.device_buttons.addButton(self.gpu_radio)
        device_layout.addWidget(self.gpu_radio)

        # GPU Info Label
        if self.device_manager.cuda_info["available"]:
            gpu_info = (
                f"GPU: {self.device_manager.cuda_info['device_name']}\n"
                f"Verfügbarer Speicher: "
                f"{(self.device_manager.cuda_info['memory_total'] - self.device_manager.cuda_info['memory_allocated']):.0f} MB"
            )
            gpu_info_label = QLabel(gpu_info)
            device_layout.addWidget(gpu_info_label)

        controls_layout.addWidget(device_group)

        # CPU-Kern-Auswahl
        cpu_layout = QHBoxLayout()
        self.label_cpu_cores = QLabel("CPU-Kerne freilassen:")
        cpu_layout.addWidget(self.label_cpu_cores)
        self.input_cpu_cores = QLineEdit("2")
        self.input_cpu_cores.setFixedWidth(50)
        cpu_layout.addWidget(self.input_cpu_cores)
        total_cores_label = QLabel(f"(Verfügbar: {multiprocessing.cpu_count()} Kerne)")
        cpu_layout.addWidget(total_cores_label)
        cpu_layout.addStretch()
        controls_layout.addLayout(cpu_layout)

        # Modellauswahl
        model_layout = QHBoxLayout()
        self.label_model = QLabel("Whisper-Modell:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS.keys())
        self.model_combo.setCurrentText([k for k, v in WHISPER_MODELS.items() if v["name"] == "small"][0])
        self.model_combo.currentTextChanged.connect(self.update_model_info)
        model_layout.addWidget(self.label_model)
        model_layout.addWidget(self.model_combo)
        controls_layout.addLayout(model_layout)

        # Modell-Info Label
        self.model_info_label = QLabel()
        self.update_model_info()
        controls_layout.addWidget(self.model_info_label)
        # Dateiauswahl und Speicherort
        file_section = QFrame()
        file_layout = QVBoxLayout(file_section)

        # Video-Auswahl
        self.label_video = QLabel("Ziehe eine Video-Datei hierher oder wähle eine aus")
        self.label_video.setStyleSheet("border: 2px dashed gray; padding: 10px;")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_layout.addWidget(self.label_video)

        video_btn_layout = QHBoxLayout()
        self.btn_select_file = QPushButton("Video auswählen")
        self.btn_select_file.clicked.connect(self.select_video)
        video_btn_layout.addWidget(self.btn_select_file)
        file_layout.addLayout(video_btn_layout)

        # Ausgabeordner
        output_layout = QHBoxLayout()
        self.label_output = QLabel("Ausgabeordner:")
        output_layout.addWidget(self.label_output)
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        output_layout.addWidget(self.output_path)
        self.btn_select_output = QPushButton("Durchsuchen")
        self.btn_select_output.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.btn_select_output)
        file_layout.addLayout(output_layout)

        # Basisname für Dateien
        filename_layout = QHBoxLayout()
        self.label_filename = QLabel("Basisname für Dateien:")
        filename_layout.addWidget(self.label_filename)
        self.input_filename = QLineEdit()
        filename_layout.addWidget(self.input_filename)
        file_layout.addLayout(filename_layout)

        controls_layout.addWidget(file_section)

        # Checkbox für WAV-Datei
        wav_layout = QHBoxLayout()
        self.keep_wav_checkbox = QCheckBox("WAV-Datei behalten")
        self.keep_wav_checkbox.setChecked(False)  # Standardmäßig nicht aktiviert
        wav_layout.addWidget(self.keep_wav_checkbox)
        file_layout.addLayout(wav_layout)

        # Fortschritt und Status
        progress_section = QFrame()
        progress_layout = QVBoxLayout(progress_section)

        # Fortschrittsbalken
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_bar)

        # Start/Abbruch Buttons
        button_layout = QHBoxLayout()
        self.btn_start = QPushButton("Transkription starten")
        self.btn_start.clicked.connect(self.start_transcription)
        button_layout.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_transcription)
        button_layout.addWidget(self.btn_cancel)
        progress_layout.addLayout(button_layout)

        # Status und Zeitinformationen
        self.label_status = QLabel("")
        progress_layout.addWidget(self.label_status)

        time_form_layout = QFormLayout()
        self.start_time_label = QLabel("Startzeit: --")
        self.end_time_label = QLabel("Endzeit: --")
        self.duration_label = QLabel("Dauer: --")
        self.remaining_time_label = QLabel("Verbleibend: --")

        time_form_layout.addRow("Startzeit:", self.start_time_label)
        time_form_layout.addRow("Endzeit:", self.end_time_label)
        time_form_layout.addRow("Dauer:", self.duration_label)
        time_form_layout.addRow("Verbleibend:", self.remaining_time_label)
        progress_layout.addLayout(time_form_layout)

        controls_layout.addWidget(progress_section)
        main_layout.addWidget(controls_container)

        self.setLayout(main_layout)

        # Media Player initialisieren
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # Weitere Initialisierungen
        self.video_path = None
        self.transcription_thread = None
        self.start_time = None
        self.end_time = None

    def update_model_info(self):
        """Aktualisiert die Modellinformationen basierend auf der aktuellen Auswahl"""
        selected_model = WHISPER_MODELS[self.model_combo.currentText()]
        memory_required = selected_model["memory"]

        if self.gpu_radio.isChecked() and self.device_manager.cuda_info["available"]:
            available_memory = (
                self.device_manager.cuda_info["memory_total"] -
                self.device_manager.cuda_info["memory_allocated"]
            )
            memory_ok = available_memory >= memory_required

            self.model_info_label.setText(
                f"Erforderlicher Speicher: {memory_required} MB\n"
                f"Verfügbarer GPU-Speicher: {available_memory:.0f} MB\n"
                f"Status: {'✓ Ausreichend' if memory_ok else '⚠ Zu wenig Speicher'}"
            )
        else:
            system_ram = psutil.virtual_memory().total / (1024 * 1024)  # MB
            memory_ok = system_ram >= memory_required * 2  # Doppelter Speicher für CPU

            self.model_info_label.setText(
                f"Erforderlicher Speicher: {memory_required * 2} MB (CPU)\n"
                f"Verfügbarer RAM: {system_ram:.0f} MB\n"
                f"Status: {'✓ Ausreichend' if memory_ok else '⚠ Zu wenig Speicher'}"
            )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0].toLocalFile()
            if url.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                event.acceptProposedAction()

    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            self.load_video(files[0])

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Video auswählen",
            "",
            "Video-Dateien (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.load_video(file_path)

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Ausgabeordner wählen",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.output_path.setText(directory)
            # Automatisch Basisnamen vorschlagen, wenn ein Video geladen ist
            if self.video_path:
                self.suggest_base_filename()

    def load_video(self, file_path):
        self.video_path = file_path
        self.label_video.setText(f"Gewählte Datei: {os.path.basename(file_path)}")
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.btn_play_pause.setEnabled(True)

        # Ausgabeordner auf Videoverzeichnis setzen, wenn noch nicht gewählt
        if not self.output_path.text():
            self.output_path.setText(os.path.dirname(file_path))

        self.suggest_base_filename()

    def suggest_base_filename(self):
        if self.video_path:
            # Basis-Dateiname aus Video-Dateinamen generieren
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            self.input_filename.setText(base_name)

    def toggle_play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.btn_play_pause.setText("Play")
        else:
            self.media_player.play()
            self.btn_play_pause.setText("Pause")

    def set_volume(self, value):
        self.audio_output.setVolume(value / 100)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if self.transcription_thread:
            remaining_time = self.transcription_thread.estimate_remaining_time(value)
            self.remaining_time_label.setText(remaining_time)

    def update_status(self, message):
        self.label_status.setText(message)

    def start_transcription(self):
        if not self.video_path:
            QMessageBox.warning(self, "Fehler", "Bitte wähle eine Video-Datei aus.")
            return

        # Überprüfen, ob die Videodatei noch existiert
        if not os.path.exists(self.video_path):
            QMessageBox.warning(
                self,
                "Fehler",
                f"Die Videodatei wurde nicht gefunden:\n{self.video_path}\n\nBitte wähle die Datei erneut aus."
            )
            self.video_path = None
            self.label_video.setText("Ziehe eine Video-Datei hierher oder wähle eine aus")
            self.btn_play_pause.setEnabled(False)
            return

        if not self.output_path.text():
            QMessageBox.warning(self, "Fehler", "Bitte wähle einen Ausgabeordner.")
            return

        base_filename = self.input_filename.text().strip()
        if not base_filename:
            QMessageBox.warning(self, "Fehler", "Bitte gib einen Basisnamen für die Dateien ein.")
            return

        try:
            num_cores_to_leave_free = int(self.input_cpu_cores.text())
            num_total_cores = multiprocessing.cpu_count()
            if num_cores_to_leave_free >= num_total_cores:
                QMessageBox.warning(
                    self,
                    "Fehler",
                    f"Die Anzahl der freizulassenden Kerne muss kleiner als die Gesamtzahl ({num_total_cores}) sein."
                )
                return
            num_cores_to_use = max(1, num_total_cores - num_cores_to_leave_free)
        except ValueError:
            QMessageBox.warning(self, "Fehler", "Bitte gib eine gültige Zahl für die freizulassenden CPU-Kerne ein.")
            return

        # Vollständiger Pfad für die Ausgabedateien
        output_base_path = os.path.join(self.output_path.text(), base_filename)

        # Überprüfen, ob Dateien bereits existieren
        existing_files = []
        for ext in ['_de.txt', '_en.txt', '.srt']:
            if os.path.exists(output_base_path + ext):
                existing_files.append(os.path.basename(output_base_path + ext))

        if existing_files:
            reply = QMessageBox.question(
                self,
                "Dateien überschreiben?",
                f"Die folgenden Dateien existieren bereits:\n{chr(10).join(existing_files)}\n\nMöchten Sie diese überschreiben?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Device-Typ bestimmen
        device_type = "cuda" if self.gpu_radio.isChecked() and self.device_manager.cuda_info["available"] else "cpu"

        # Modell-Speicheranforderungen prüfen
        selected_model = WHISPER_MODELS[self.model_combo.currentText()]
        if device_type == "cuda":
            available_memory = (
                self.device_manager.cuda_info["memory_total"] -
                self.device_manager.cuda_info["memory_allocated"]
            )
            if available_memory < selected_model["memory"]:
                QMessageBox.warning(
                    self,
                    "Warnung",
                    f"Nicht genügend GPU-Speicher verfügbar!\n"
                    f"Benötigt: {selected_model['memory']} MB\n"
                    f"Verfügbar: {available_memory:.0f} MB"
                )
                return

        self.start_time = datetime.now()
        self.start_time_label.setText(f"Startzeit: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.end_time_label.setText("Endzeit: --")
        self.duration_label.setText("Dauer: --")
        self.remaining_time_label.setText("Verbleibend: --")

        # Buttons aktualisieren
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # Thread starten (nur einmal!)
        self.transcription_thread = TranscriptionWorker(
            self.video_path,
            output_base_path,
            self.model_combo.currentText(),
            device_type,
            num_cores_to_use=num_cores_to_use if device_type == "cpu" else None,
            keep_wav=self.keep_wav_checkbox.isChecked()
        )

        self.transcription_thread.progress_signal.connect(self.update_progress)
        self.transcription_thread.status_signal.connect(self.update_status)
        self.transcription_thread.finished_signal.connect(self.transcription_finished)
        self.transcription_thread.error_signal.connect(self.transcription_error)
        self.transcription_thread.time_estimate_signal.connect(
            lambda t: self.remaining_time_label.setText(f"Verbleibend: {t}")
        )

        self.transcription_thread.start()

    def cancel_transcription(self):
        if self.transcription_thread and self.transcription_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Transkription abbrechen",
                "Möchten Sie die Transkription wirklich abbrechen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.transcription_thread.cancel()
                self.btn_cancel.setEnabled(False)
                self.label_status.setText("Breche Transkription ab...")

    def transcription_finished(self, message):
        self.end_time = datetime.now()
        self.end_time_label.setText(f"Endzeit: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        duration = self.end_time - self.start_time
        self.duration_label.setText(f"Dauer: {str(duration).split('.')[0]}")
        self.remaining_time_label.setText("Verbleibend: --")

        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        # Erfolgreiche Fertigstellung
        QMessageBox.information(self, "Fertig", message)
        self.label_status.setText("Bereit")
        self.progress_bar.setValue(0)

    def transcription_error(self, error_message):
        self.end_time = datetime.now()
        self.end_time_label.setText(f"Endzeit: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        duration = self.end_time - self.start_time
        self.duration_label.setText(f"Dauer: {str(duration).split('.')[0]}")
        self.remaining_time_label.setText("Verbleibend: --")

        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        # Fehlermeldung anzeigen
        QMessageBox.critical(self, "Fehler", error_message)
        self.label_status.setText("Fehler aufgetreten")
        self.progress_bar.setValue(0)

    def closeEvent(self, event):
        """Wird aufgerufen, wenn das Fenster geschlossen wird"""
        if self.transcription_thread and self.transcription_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Transkription läuft",
                "Eine Transkription läuft noch. Möchten Sie diese abbrechen und das Programm beenden?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.transcription_thread.cancel()
                self.transcription_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    # Hochauflösende Displays unterstützen
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Style für die Anwendung setzen
    app.setStyle('Fusion')

    # Fenster erstellen und anzeigen
    window = TranscriptionApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
