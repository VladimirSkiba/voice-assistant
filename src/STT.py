from abc import ABC, abstractmethod
import numpy as np
import json
import os
import tempfile
import time
from vosk import Model, KaldiRecognizer, SetLogLevel
import soundfile as sf


SAMPLE_RATE = 16000


# ─── Абстрактный интерфейс ────────────────────────────────────────────────────

class Transcriber(ABC):
    """Абстрактный интерфейс для всех транскрайберов"""
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """Транскрибирует аудио в текст"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Возвращает название движка"""
        pass




class WhisperTranscriber(Transcriber):
    def __init__(self, client, language: str):
        self.client = client          
        self.language = language      

    def transcribe(self, audio: np.ndarray) -> str:  
        if len(audio) == 0:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            print(f"[DEBUG] длительность: {len(audio)/SAMPLE_RATE:.1f}с  |  отправляю в Whisper...")

            t0 = time.monotonic()
            with open(tmp.name, "rb") as f:
                # Используем self.client и self.language
                result = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=f, 
                    language=self.language
                )
            print(f"[DEBUG] ответ за {time.monotonic()-t0:.1f}с: '{result.text.strip()}'")

        os.unlink(tmp.name)
        return result.text.strip()

    def get_name(self) -> str:
        return "Whisper API"




class Vosk22Transcriber(Transcriber):
    def __init__(self, model_path: str = "models/vosk-model-small-ru-0.22"):
        print("[INFO] Загрузка модели Vosk в память...")
        SetLogLevel(-1)
        self.vosk_model = Model(model_path)
        self.sample_rate = 16000
        
        print("[INFO] Модель Vosk загружена.")

    def transcribe(self, audio: np.ndarray) -> str:  
        if len(audio) == 0:
            return ""

        current_sr = SAMPLE_RATE
        if SAMPLE_RATE != self.sample_rate:
            from scipy.signal import resample
            num_samples = int(len(audio) * self.sample_rate / SAMPLE_RATE)
            audio = resample(audio, num_samples)
            current_sr = self.sample_rate

        # Конвертация в 16-bit PCM
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)

        print(f"[DEBUG] длительность: {len(audio)/current_sr:.1f}с  |  отправляю в Vosk...")

        t0 = time.monotonic()
        
        # ← Используем self.vosk_model
        rec = KaldiRecognizer(self.vosk_model, current_sr)
        
        pcm_bytes = audio.tobytes()
        rec.AcceptWaveform(pcm_bytes)
        
        final_result = json.loads(rec.FinalResult())
        text = final_result.get("text", "").strip()
        
        print(f"[DEBUG] ответ за {time.monotonic()-t0:.1f}с: '{text}'")
        return text

    def get_name(self) -> str:
        return "Vosk (локальная модель)"


class VoskTranscriber(Transcriber):
    def __init__(self, model_path: str = "models/vosk-model-small-ru-0.2"):
        print("[INFO] Загрузка модели Vosk в память...")
        SetLogLevel(-1)
        
        self.vosk_model = Model(model_path)
        self.sample_rate = 16000
        
        print("[INFO] Модель Vosk загружена.")

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) == 0:
            return ""

        current_sr = SAMPLE_RATE
        # Ресемплинг, если частота исходного аудио отличается от 16000 Гц
        if SAMPLE_RATE != self.sample_rate:
            try:
                from scipy.signal import resample
                num_samples = int(len(audio) * self.sample_rate / SAMPLE_RATE)
                audio = resample(audio, num_samples)
                current_sr = self.sample_rate
            except ImportError:
                print("[WARNING] scipy не установлен. Ресемплинг невозможен. Убедитесь, что аудио уже 16000 Гц.")

        # Конвертация в 16-bit PCM (int16)
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)

        print(f"[DEBUG] длительность: {len(audio)/current_sr:.1f}с | отправляю в Vosk...")

        t0 = time.monotonic()
        rec = KaldiRecognizer(self.vosk_model, current_sr)
        
        pcm_bytes = audio.tobytes()
        rec.AcceptWaveform(pcm_bytes)
        
        final_result = json.loads(rec.FinalResult())
        text = final_result.get("text", "").strip()
        
        print(f"[DEBUG] ответ за {time.monotonic()-t0:.1f}с: '{text}'")
        return text

    def get_name(self) -> str:
        return "Vosk (локальная модель)"

class TranscriberFactory:
    """Фабрика для создания транскрайберов."""

    @staticmethod
    def create(engine_type: str, **kwargs) -> Transcriber:
        if engine_type == "whisper":
            return WhisperTranscriber(
                client=kwargs.get("client"),
                language=kwargs.get("language", "ru")
            )
        elif engine_type == "vosk22":
            return Vosk22Transcriber(
                model_path=kwargs.get("model_path", "models/vosk-model-small-ru-0.22")
            )
        else:
            raise ValueError(f"Неизвестный тип движка: {engine_type}")


def select_transcriber(client) -> Transcriber:
    """Интерактивный выбор движка распознавания речи."""

    print("\n=== Выбор движка распознавания речи ===")
    print("1. Whisper")
    print("2. Vosk22 (маленькая локальная модель)")
    print("3. Vosk42 (маленькая локальная модель)")


    while True:
        choice = input("\nВведите номер (1-3): ").strip()

        if choice == "1":
            return TranscriberFactory.create(
                "whisper",
                client=client,
                language="ru"
            )

        if choice == "2":
            model_path = input(
                "Путь к папке с моделью Vosk [models/vosk-model-small-ru-0.22]: "
            ).strip()

            if not model_path:
                model_path = "models/vosk-model-small-ru-0.22"

            if not os.path.exists(model_path):
                print(f"[!] Папка не найдена: {model_path}")
                continue

            return TranscriberFactory.create(
                "vosk22",
                model_path=model_path
            )

        if choice == "3":
            model_path = input(
                "Путь к папке с моделью Vosk [models/vosk-model-small-ru-0.42]: "
            ).strip()

            if not model_path:
                model_path = "models/vosk-model-small-ru-0.42"

            if not os.path.exists(model_path):
                print(f"[!] Папка не найдена: {model_path}")
                continue

            return TranscriberFactory.create(
                "vosk42",
                model_path=model_path
            )

        print("[!] Неверный выбор, попробуйте снова")