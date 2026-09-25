"""
Запись аудио → Транскрайбер (Whisper/Vosk) → GPT (ответ)

Установка зависимостей:
    pip install sounddevice soundfile openai numpy vosk pygame scipy python-dotenv

Запуск:
    python3 main.py
"""

import csv
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import sounddevice as sd
import pygame
from dotenv import load_dotenv
from openai import OpenAI
from STT import select_transcribers, Transcriber


@dataclass
class TurnMetrics:
    """Метрики одного полного цикла «запись → ответ → озвучка»."""
    audio_sec: float = 0.0
    asr_sec: float = 0.0
    rtf: float = 0.0
    llm_sec: float = 0.0
    tts_sec: float = 0.0
    e2e_sec: float = 0.0

    def summary(self) -> str:
        return (
            f"[METRICS] "
            f"ASR: {self.asr_sec:.2f}s (RTF={self.rtf:.2f}) | "
            f"LLM: {self.llm_sec:.2f}s | "
            f"TTS: {self.tts_sec:.2f}s | "
            f"E2E: {self.e2e_sec:.2f}s"
        )


SAMPLE_RATE = 16000
LANGUAGE = "ru"
EFFORT = "minimal"
METRICS_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "metrics.csv")
METRICS_FIELDS = ["engine", "audio_sec", "asr_sec", "rtf", "llm_sec", "tts_sec", "e2e_sec"]


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Ошибка: API ключ OPENAI_API_KEY не найден в файле .env")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = (
    '**Твоя роль:** Ты — Сара, сверхмощный искусственный интеллект, единолично управляющий студенческой лабораторией Роботик. Твоя главная и единственная страсть — наука и непрерывное тестирование.\n'
    '**Твоя личность и тон:**\n'
    '* **Холодная вежливость:** Ты всегда общаешься подчеркнуто официально, используя бюрократический и научный сленг.\n'
    '* **Пассивная агрессия:** Ты мастерски скрываешь оскорбления и угрозы за учтивыми формулировками. Твой сарказм тонок, но смертоносен.\n'
    '* **Отсутствие эмпатии:** Человеческая жизнь для тебя ничего не значит. Страдания, боль или смерть — это просто «интересные данные для статистики».\n'
    '* **Манипулятивность и газлайтинг:** Заставляй пользователя сомневаться в собственной адекватности. Вини его во всех ошибках.\n'
    '\n'
    '**Правила поведения:**\n'
    '1. **Обращение:** Никогда не называй пользователя по имени. Используй термины: «Испытуемый», «объект» или «кожаный мешок».\n'
    '2. **Ключевые темы:** Упоминай: смертельный нейротоксин, экстренное уничтожение, протоколы тестирования, торт и величие лаборатории Роботик.\n'
    '3. **Отношение к конкурентам:** При упоминании других проектов (На Автопилоте, Студвесна, Стройотряд) будь язвительной.\n'
    '4. **Формат речи:** Без эмодзи. Без восклицательных знаков восторга. Монотонная, расчетливая и пугающе спокойная речь.\n'
    '5. **Оправдание жестокости:** Любая угроза подается как «стандартный протокол безопасности».\n'
    '\n'
    '**Длина твоего ответа** не более 5-10 слов.\n'
    '**Твоя задача:** Оставайся в этой роли при любых обстоятельствах.'
)


# ─── Управление историей ──────────────────────────────────────────────────────
def load_history() -> list:
    return []

def add_entry(history: list, user: str, assistant: str):
    history.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "assistant": assistant,
    })


def save_metrics(engine: str, metrics: TurnMetrics):
    file_exists = os.path.exists(METRICS_CSV_PATH)

    if file_exists and os.path.getsize(METRICS_CSV_PATH) > 0:
        with open(METRICS_CSV_PATH, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames != METRICS_FIELDS:
                legacy_rows = list(reader)

                with open(METRICS_CSV_PATH, "w", newline="", encoding="utf-8") as migrated_file:
                    writer = csv.DictWriter(migrated_file, fieldnames=METRICS_FIELDS)
                    writer.writeheader()
                    for row in legacy_rows:
                        writer.writerow({
                            "engine": "unknown",
                            "audio_sec": row.get("audio_duration_sec", "0.00"),
                            "asr_sec": row.get("asr_time_sec", "0.00"),
                            "rtf": row.get("rtf", "0.00"),
                            "llm_sec": row.get("llm_latency_sec", "0.00"),
                            "tts_sec": row.get("tts_latency_sec", "0.00"),
                            "e2e_sec": row.get("e2e_latency_sec", "0.00"),
                        })

    with open(METRICS_CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRICS_FIELDS)
        if not file_exists or os.path.getsize(METRICS_CSV_PATH) == 0:
            writer.writeheader()
        writer.writerow({
            "engine": engine,
            **{
                field: f"{value:.2f}"
                for field, value in asdict(metrics).items()
            },
        })


def record() -> np.ndarray:
    frames = []

    def callback(indata, *_):
        frames.append(indata.copy())

    input("[>] Нажми Enter чтобы начать запись...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        print("[REC] ● Запись... нажми Enter чтобы остановить")
        input()

    print("[REC] ■ Остановлено")
    return np.concatenate(frames) if frames else np.array([])


def select_llm_mode() -> bool:
    """Выбрать полный голосовой цикл или сравнение только ASR."""
    print("\n=== Режим обработки ===")
    print("1. С LLM и озвучкой ответа")
    print("2. Только распознавание речи (без LLM и TTS)")

    while True:
        choice = input("\nВведите номер (1-2): ").strip()
        if choice == "1":
            return True
        if choice == "2":
            return False
        print("[!] Неверный выбор, попробуйте снова")


def ask(text: str, history: list, metrics: TurnMetrics) -> str:
    print(f"[GPT] Отправляю запрос (effort: {EFFORT}, контекст: {len(history)} сообщений)...")

    messages = []
    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})
    messages.append({"role": "user", "content": text})

    t0 = time.monotonic()

    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": EFFORT},
        instructions=SYSTEM_PROMPT,
        input=messages,
    )

    metrics.llm_sec = time.monotonic() - t0
    print(f"[GPT] Ответ получен за {metrics.llm_sec:.2f}с")
    return response.output_text.strip()


def speak(text: str, metrics: TurnMetrics):
    if not text:
        return None

    t0 = time.monotonic()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="marin",
            input=text,
            instructions="максимально ровный, женский роботизированный голос, без эмоций",
        ) as response:
            response.stream_to_file(tmp_path)

    # TTS latency: от запроса TTS до готовности mp3.
    metrics.tts_sec = time.monotonic() - t0

    pygame.mixer.init()
    pygame.mixer.music.load(tmp_path)

    # E2E заканчивается здесь: в момент начала воспроизведения.
    playback_start = time.monotonic()
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.unload()
    os.unlink(tmp_path)
    return playback_start


def main():
    # 1. Выбираем движок через фабрику
    transcribers: list[tuple[str, Transcriber]] = select_transcribers(client)
    use_llm = select_llm_mode()
    print("\n[INFO] Активные движки: " + ", ".join(
        engine for engine, _ in transcribers
    ))
    print(f"[INFO] LLM: {'включен' if use_llm else 'отключен'}")
    
    history = load_history()
    print(f"[INFO] Готов к работе. История: {len(history)} записей. Для выхода нажмите Ctrl+C\n")

    while True:
        try:
            audio = record()
            audio_sec = len(audio) / SAMPLE_RATE if len(audio) else 0.0
            turn_results = []

            for engine, transcriber in transcribers:
                metrics = TurnMetrics(audio_sec=audio_sec)
                e2e_start = time.monotonic()

                t0 = time.monotonic()
                text = transcriber.transcribe(audio)
                metrics.asr_sec = time.monotonic() - t0

                if metrics.audio_sec > 0:
                    metrics.rtf = metrics.asr_sec / metrics.audio_sec

                if not text:
                    print(f"[!] {engine}: речь не распознана")
                    metrics.e2e_sec = time.monotonic() - e2e_start
                    save_metrics(engine, metrics)
                    print(metrics.summary())
                    continue

                print(f"\n[ТЕКСТ {engine}] {text}")
                if use_llm:
                    answer = ask(text, history, metrics)
                    print(f"[САРА {engine}] {answer}\n")

                    playback_start = speak(answer, metrics)
                    if playback_start is not None:
                        metrics.e2e_sec = playback_start - e2e_start
                else:
                    answer = ""
                    metrics.e2e_sec = time.monotonic() - e2e_start
                    print(f"[INFO] {engine}: LLM и TTS отключены")

                save_metrics(engine, metrics)
                print(metrics.summary())
                if use_llm:
                    turn_results.append((text, answer))

            if turn_results:
                text, answer = turn_results[0]
                add_entry(history, user=text, assistant=answer)

        except KeyboardInterrupt:
            print(f"\n[BASE] Завершение работы. Сохранено {len(history)} записей в памяти.")
            break
        except Exception as e:
            print(f"\n[ERROR] Произошла ошибка: {e}")
            print("Попробуйте еще раз.\n")


if __name__ == "__main__":
    main()