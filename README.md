# Voice Assistant for Speech Recognition Research

Проект голосового помощника для исследования потокового распознавания речи, задержек и качества диалоговых систем в рамках научно-исследовательской работы (НИР).

## Описание

Данный проект представляет собой модульную систему голосового взаимодействия, разработанную для проведения исследований в области:
- **Задержки распознавания речи** (latency measurement)
- **Качества транскрипции** (WER — Word Error Rate анализ)
- **Сравнения облачных и локальных STT-движков**
- **Оптимизации диалоговых систем реального времени**

### Архитектура системы

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   Микрофон  │ ──▶ │   STT Engine │ ──▶ │  GPT/LMM    │ ──▶ │    TTS      │
│  (запись)   │     │ (Whisper/Vosk)│     │  (ответ)    │     │  (озвучка)  │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
```

**Поддерживаемые движки распознавания:**
- **Whisper API** (OpenAI) — облачное распознавание
- **Vosk** (локально) — офлайн-распознавание с моделями:
  - `vosk-model-small-ru-0.22` (лёгкая модель)
  - `vosk-model-ru-0.42` (полная модель)

---

## Быстрый старт

### 1. Создание виртуального окружения

```bash
python -m venv .venv
```

### 2. Установка системных зависимостей

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install libportaudio2 python3-pyaudio
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
```powershell
# PyAudio устанавливается вместе с requirements.txt
# При ошибках скачайте wheel-файл с https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

### 3. Активация виртуального окружения

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
```

### 4. Установка Python-зависимостей

```bash
pip install -r requirements.txt
```

### 5. Загрузка моделей Vosk

```bash
mkdir -p models
cd models

# Лёгкая модель (рекомендуется для тестирования задержек)
wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip

# Полная модель (для сравнения качества)
wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip
unzip vosk-model-ru-0.42.zip

# Очистка архивов
rm *.zip
cd ..
```

Примечание: Полный список моделей: [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)

### 6. Настройка API ключа

Создайте файл `.env` в корне проекта:

```bash
echo 'OPENAI_API_KEY="sk-your-api-key-here"' > .env
```

Или вручную:
```env
OPENAI_API_KEY="paste_your_api_key"
```

Получить ключ можно на: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 7. Запуск помощника

```bash
python src/main.py
```

---

## Использование для НИР

### Измерение задержки (Latency)

Система автоматически логирует время обработки на каждом этапе:

```
[DEBUG] длительность: 3.2с  |  отправляю в Vosk...
[DEBUG] ответ за 0.8с: 'привет как дела'
[GPT] Отправляю запрос (effort: minimal, контекст: 2 сообщений)...
[GPT] Ответ получен за 1.2с
```

**Метрики для сбора:**
| Метрика | Описание | Где измеряется |
|---------|----------|----------------|
| `T_record` | Длительность записи аудио | `main.py:record()` |
| `T_stt` | Время распознавания речи | `STT.py:transcribe()` |
| `T_gpt` | Время генерации ответа LLM | `main.py:ask()` |
| `T_tts` | Время синтеза речи | `main.py:speak()` |
| `T_total` | Общая задержка диалога | Сумма всех этапов |

### Сравнение движков STT

Для переключения между движками запустите `main.py` и выберите:
- **1** — Whisper API (облачный)
- **2** — Vosk small (локальный, быстрый)
- **3** — Vosk full (локальный, точный)

**Рекомендуемый протокол исследования:**
1. Записать 50+ тестовых фраз разной длины
2. Прогнать через каждый движок
3. Рассчитать WER (Word Error Rate)
4. Сравнить среднюю задержку

### Пример сбора данных

```python
# Добавьте в main.py перед transcribe()
start_time = time.monotonic()
text = transcriber.transcribe(audio)
stt_latency = time.monotonic() - start_time
print(f"[METRIC] STT Latency: {stt_latency:.3f}s")
```

---

## Структура проекта

```
voice-assistant/
├── src/
│   ├── main.py          # Основной цикл: запись → STT → GPT → TTS
│   └── STT.py           # Модуль транскрипции (интерфейсы + реализации)
├── models/              # Папка с моделями Vosk
│   ├── vosk-model-small-ru-0.22/
│   └── vosk-model-ru-0.42/
├── .env                 # API ключи (не коммитить!)
├── requirements.txt     # Зависимости Python
├── README.md            # Документация
└── LICENSE              # Лицензия
```

---

## Конфигурация

### Параметры записи аудио

В `main.py`:
```python
SAMPLE_RATE = 16000    # Частота дискретизации (Гц)
LANGUAGE = "ru"        # Язык распознавания
EFFORT = "minimal"     # Уровень усилий GPT (minimal/low/medium/high)
```

### Системный промпт

Персонаж «Сара» настроен в `SYSTEM_PROMPT` (строка 37-54).  
Для исследований можно изменить на нейтральный промпт:

```python
SYSTEM_PROMPT = "Ты полезный ассистент. Отвечай кратко и точно."
```

---

## Требования

- **Python:** 3.8+
- **ОС:** Linux / macOS / Windows
- **Микрофон:** любой, поддерживаемый ОС
- **Интернет:** требуется для Whisper API и GPT

### Зависимости

| Пакет | Назначение |
|-------|------------|
| `sounddevice` | Запись аудио с микрофона |
| `vosk` | Локальное распознавание речи |
| `openai` | Whisper API + GPT + TTS |
| `pygame` | Воспроизведение аудио-ответов |
| `numpy` | Обработка аудиоданных |
| `scipy` | Ресемплинг аудио |
| `python-dotenv` | Загрузка переменных окружения |

---

## Идеи для исследований

1. **Сравнение latency**: Whisper API vs Vosk (local)
2. **Влияние длины фразы** на точность распознавания
3. **WER анализ** на русском языке для разных моделей
4. **Оптимизация буферизации** для потокового распознавания
5. **Субъективная оценка качества** диалога (MOS тест)

---

## Вклад в проект

1. Fork репозиторий
2. Создайте ветку (`git checkout -b feature/research-metrics`)
3. Commit изменения (`git commit -m 'Add latency metrics'`)
4. Push (`git push origin feature/research-metrics`)
5. Откройте Pull Request

---

## Лицензия

MIT License — см. файл [LICENSE](LICENSE)

---

## Полезные ссылки

- [Vosk Documentation](https://alphacephei.com/vosk/)
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
- [Word Error Rate Calculator](https://github.com/jitsi/jitsi-meet/tree/master/resources/wer)
- [Научные статьи по STT](https://scholar.google.com/scholar?q=speech+recognition+latency+dialog+systems)

---

## Автор

Проект создан для учебно-исследовательской работы в вузе.

**Тема НИР:** *Исследование потокового распознавания речи на задержку и качество диалоговых систем*