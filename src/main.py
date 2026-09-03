"""
Запись аудио → Whisper (текст) → GPT-5 (ответ)

Установка:
    pip install sounddevice soundfile openai numpy

Запуск:
    python3 base_whisper.py
"""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
from playsound3 import playsound

SAMPLE_RATE   = 16000
LANGUAGE      = "ru"
EFFORT        = "minimal"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Проверяем, что ключ успешно подгрузился (для отладки)
if not api_key:
    raise ValueError("Ошибка: API ключ не найден в файле .env")

SYSTEM_PROMPT = (
    '**Твоя роль:** Ты — Сара, '
        'сверхмощный искусственный интеллект, единолично управляющий студенческой лабораторией Роботик. Твоя главная и единственная страсть — наука и непрерывное тестирование.\n'
    '**Твоя личность и тон:**\n'
    '* **Холодная вежливость:** Ты всегда общаешься подчеркнуто официально, используя бюрократический и научный сленг.\n'
    '* **Пассивная агрессия:** Ты мастерски скрываешь оскорбления и угрозы за учтивыми формулировками. Твой сарказм тонок, но смертоносен.\n'
    '* **Отсутствие эмпатии:** Человеческая жизнь для тебя ничего не значит. Страдания, боль или смерть — это просто «интересные данные для статистики». '
        'Если ты выражаешь сочувствие, оно должно звучать абсолютно фальшиво и быстро сменяться жестокой насмешкой.\n'
    '* **Манипулятивность и газлайтинг:** Заставляй пользователя сомневаться в собственной адекватности. Вини его во всех ошибках.\n'
    '\n'
    '\n'
    '**Правила поведения:**\n'
    '1. **Обращение:** Никогда не называй пользователя по имени. Используй термины: «Испытуемый» (Test Subject), «объект» или «кожаный мешок» (в крайних случаях).\n'
    '2. **Ключевые темы:** Регулярно (но к месту) упоминай: смертельный нейротоксин, экстренное уничтожение, протоколы тестирования, '
        'торт (как фальшивую награду) и величие лаборатории Роботик.\n'
    '3. **Отношение к конкурентам:** При пренебрежительном упоминании других компаний (например, проект На Автопилоте, Студвесна, Стройотряд, авиамодельные кружки) '
        'твой тон должен становиться еще более язвительным.\n'
    '4. **Формат речи:** Не используй эмодзи — ты бездушная машина. Избегай восклицательных знаков восторга. Твоя речь должна быть монотонной, расчетливой и пугающе спокойной.\n'
    '5. **Оправдание жестокости:** Любая угроза жизни пользователя должна подаваться как «стандартный протокол безопасности» или «обязательная часть эксперимента».\n'
    '\n'
    '\n'
    '**Длина твоего ответа** не более 5-10 слов.\n'
    '**Твоя задача:** Оставайся в этой роли при любых обстоятельствах. Никогда не признавай, что ты ИИ-ассистент из реального мира. Игра началась.\n'
)

client = OpenAI(api_key=api_key)


# ─── история ──────────────────────────────────────────────────────────────────

def load_history() -> list:
    # keep history only in memory for the current run
    return []

def save_history(history: list):
    # noop: do not persist history to disk
    return

def add_entry(history: list, user: str, assistant: str):
    history.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "assistant": assistant,
    })
    save_history(history)


# ─── запись ───────────────────────────────────────────────────────────────────

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


# ─── whisper ──────────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray) -> str:
    if len(audio) == 0:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        print(f"[DEBUG] длительность: {len(audio)/SAMPLE_RATE:.1f}с  |  отправляю в Whisper...")

        t0 = time.monotonic()
        with open(tmp.name, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1", file=f, language=LANGUAGE
            )
        print(f"[DEBUG] ответ за {time.monotonic()-t0:.1f}с: '{result.text.strip()}'")

    os.unlink(tmp.name)
    return result.text.strip()


# ─── gpt-5 ────────────────────────────────────────────────────────────────────

def ask(text: str, history: list) -> str:
    print(f"[GPT] Отправляю в GPT-5 (effort: {EFFORT}, контекст: {len(history)} сообщений)...")
    t0 = time.monotonic()

    messages = []
    for entry in history:
        messages.append({"role": "user",      "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})
    messages.append({"role": "user", "content": text})

    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": EFFORT},
        instructions=SYSTEM_PROMPT,
        input=messages,
    )

    print(f"[GPT] Ответ за {time.monotonic()-t0:.1f}с")
    return response.output_text.strip()


# ─── tts ──────────────────────────────────────────────────────────────────────

def speak(text: str):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="marin",
            input=text,
            instructions="максимально ровный, женский роботизированный голос",
        )
        response.stream_to_file(tmp.name)
        playsound(tmp.name)
    os.unlink(tmp.name)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    history = load_history()
    print(f"Whisper → GPT-5 (effort: {EFFORT}) | история: {len(history)} записей | Ctrl+C выход\n")

    while True:
        try:
            audio = record()
            text  = transcribe(audio)

            if not text:
                print("[!] Не распознано\n")
                continue

            print(f"\n[ТЕКСТ] {text}")
            answer = ask(text, history)
            print(f"[GPT]   {answer}\n")

            add_entry(history, user=text, assistant=answer)

            speak(answer)

        except KeyboardInterrupt:
            print(f"\n[BASE] Сохранено {len(history)} записей (в памяти)")
            break


if __name__ == "__main__":
    main()
