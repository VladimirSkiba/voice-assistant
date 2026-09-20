# main.py
"""
Запись аудио → Транскрайбер (Whisper/Vosk) → GPT (ответ)

Установка зависимостей:
    pip install sounddevice soundfile openai numpy vosk pygame scipy python-dotenv

Запуск:
    python3 main.py
"""

import os
import tempfile
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import pygame
from dotenv import load_dotenv
from openai import OpenAI
from STT import select_transcriber, Transcriber

SAMPLE_RATE = 16000
LANGUAGE = "ru"
EFFORT = "minimal"


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


def ask(text: str, history: list) -> str:
    print(f"[GPT] Отправляю запрос (effort: {EFFORT}, контекст: {len(history)} сообщений)...")
    t0 = time.monotonic()

    messages = []
    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})
    messages.append({"role": "user", "content": text})


    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": EFFORT},
        instructions=SYSTEM_PROMPT,
        input=messages,
    )

    print(f"[GPT] Ответ получен за {time.monotonic()-t0:.1f}с")
    return response.output_text.strip()


def speak(text: str):
    if not text:
        return
        
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="marin", # или "alloy", "shimmer" в зависимости от доступных
            input=text,
            instructions="максимально ровный, женский роботизированный голос, без эмоций",
        )
        response.stream_to_file(tmp.name)
        
        pygame.mixer.init()
        pygame.mixer.music.load(tmp.name)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    os.unlink(tmp.name)



def main():
    # 1. Выбираем движок через фабрику
    transcriber: Transcriber = select_transcriber(client)
    print(f"\n[INFO] Активный движок распознавания: {transcriber.get_name()}")
    
    history = load_history()
    print(f"[INFO] Готов к работе. История: {len(history)} записей. Для выхода нажмите Ctrl+C\n")

    while True:
        try:
            audio = record()
            

            text = transcriber.transcribe(audio)

            if not text:
                print("[!] Не распознано. Попробуйте еще раз.\n")
                continue

            print(f"\n[ТЕКСТ] {text}")
            
            answer = ask(text, history)
            print(f"[САРА] {answer}\n")

            add_entry(history, user=text, assistant=answer)
            speak(answer)

        except KeyboardInterrupt:
            print(f"\n[BASE] Завершение работы. Сохранено {len(history)} записей в памяти.")
            break
        except Exception as e:
            print(f"\n[ERROR] Произошла ошибка: {e}")
            print("Попробуйте еще раз.\n")


if __name__ == "__main__":
    main()