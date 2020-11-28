# Telegram бот "Memes 2020"

Бот умеет:
- генерировать мемы с заданной периодичностью
- генерировать мем по запросу пользователя
- генерировать мем по заданному началу текста

Для запуска желательно наличие gpu с поддержкой CUDA.

### Инструкция по установке

Общие шаги по установке:

1. Поместить файлы моделей в папки `<имя_мема>/model/checkpoint/prod`

Запуск без Docker

1. Установить CUDA 10.0
2. Установить пакеты Python
    - tensorflow==1.15
    - gpt-2-simple
    - pillow
    - pandas
    - python-telegram-bot
3. Выполнить `python ./src/bot.py`

Запуск с Docker

1. Создать образ Docker `sudo docker build -t bot_image .`
2. Развернуть образ Docker `docker run -d --gpus=all bot_image`
