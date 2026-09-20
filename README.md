# Voice-Assistant

* ## 1 Создаём виртуальное окружение 
    ```
    python -m venv .venv
    ```
* ## 1.1 Устанавливаем системную библиотеку для записи аудио
    ```
    sudo apt update
    sudo apt install libportaudio2
    ```
*   ## 2 Активируем виртуальное окружение 
    ```
    source .venv/bin/activate
    ```
*   ## 3 Устанавливаем бибиотеки
    ```
    pip install -r requirements.txt
    ```
*   ## 4 Создаём папку `models` и скачиваем модели Vosk
    ```
    mkdir -p models
    cd models
    wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
    wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip
    unzip vosk-model-small-ru-0.22.zip
    unzip vosk-model-ru-0.42.zip
    rm vosk-model-small-ru-0.22.zip vosk-model-ru-0.42.zip
    cd ..
    ```
    Список моделей: [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
    ## 5 Создаём файл .env
    который внутри должен содержать ваш API
    ```
    OPENAI_API_KEY="paste_your_api"
    ```
*   ## 6 Запускаем помошника
    ```
    python src/main.py
    ```