# Образ: Ubuntu + Python 3.11 + базовые dev-тулзы от MS
FROM mcr.microsoft.com/devcontainers/python:3.11

USER root

# Установка системных зависимостей для работы со звуком
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        ffmpeg \
        libasound2 \
        libportaudio2 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# Возвращаемся к пользователю vscode (стандарт для образов MS)
USER vscode