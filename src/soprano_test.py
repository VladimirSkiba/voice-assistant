"""
Озвучка одной строки: Soprano TTS → сохранение в WAV → воспроизведение через pygame

Установка:
    pip install soprano-tts pygame
"""

import os
import tempfile
import pygame
from soprano import SopranoTTS


# ═══════════════════════════════════════════════════════════
# 1. Инициализация модели (один раз при запуске)
# ═══════════════════════════════════════════════════════════
print("[1/3] Загрузка модели Soprano...")
tts_model = SopranoTTS(device="auto", backend="auto")
print("[OK] Модель загружена.\n")


# ═══════════════════════════════════════════════════════════
# 2. Текст для озвучки
# ═══════════════════════════════════════════════════════════
text_to_speak = "Привет. Это тестовая строка, озвученная локально через Soprano и pygame."


# ═══════════════════════════════════════════════════════════
# 3. Генерация WAV + воспроизведение через pygame
# ═══════════════════════════════════════════════════════════
def speak(text: str):
    print(f"[TTS] Синтез: '{text}'")
    
    # Создаем временный файл (автоматически удалится после использования)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Генерируем аудио и сохраняем в файл
        tts_model.infer(text, tmp_path)
        
        # Инициализируем pygame микшер (если ещё не инициализирован)
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Загружаем и проигрываем
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        
        print("[TTS] Воспроизведение...")
        
        # Ждём, пока трек не закончится
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        print("[OK] Воспроизведение завершено.")
        
    finally:
        # Удаляем временный файл
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Запуск
print("[2/3] Запуск озвучки...")
speak(text_to_speak)

# Корректно завершаем работу pygame
pygame.mixer.quit()
print("[3/3] Готово.")