#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Nov 16 10:29:44 2020

@author: atta
"""

# pylint: disable=W0613, C0116
# type: ignore[union-attr]

import logging
import json

# from telegram import Update,
# from telegram.ext import Updater, CommandHandler

from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

from memeprovider import MemeProvider

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

BOT_STATE_INITIAL, BOT_STATE_TYPING_REPLY = range(2)  
    
reply_keyboard = [
    ['Получить мем'],
    ['Ввести начало мема'],
]

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Используйте команду /set <seconds>, чтобы запустить постинг мемов по времени', \
                              reply_markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False))
    return BOT_STATE_INITIAL


def post_meme(context):
    job = context.job
    job_context = job.context
    chat_id = job_context['chat_id']
    meme_provider = job_context['meme_provider']
    img = meme_provider.get_next()
    try:
        if img is None:
            context.bot.send_message(chat_id, 'Не удалось получить мем: новые мемы в процессе генерации') 
        else:
            context.bot.send_photo(chat_id, img)
    except Exception as e:
        job.schedule_removal()
        raise e

def remove_job_if_exists(name, context):
    current_jobs = context.job_queue.get_jobs_by_name(name)
    if not current_jobs:
        return False
    for job in current_jobs:
        job.schedule_removal()
    return True


def set_timer(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    try:
        due = int(context.args[0])
        if due < 0:
            update.message.reply_text('Неверное значение')
            return

        job_removed = remove_job_if_exists(str(chat_id), context)
        job_context = {
            'chat_id': chat_id,
            'meme_provider': context.dispatcher.user_data['meme_provider']
            }
        context.job_queue.run_repeating(post_meme, due, context=job_context, name=str(chat_id))

        text = 'Постинг запущен'
        if job_removed:
            text += ' Предыдущий постинг отключен'
        update.message.reply_text(text)

    except (IndexError, ValueError):
        update.message.reply_text('Использование: /set <seconds>')
    
    return BOT_STATE_INITIAL


def unset(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    job_removed = remove_job_if_exists(str(chat_id), context)
    text = 'Постинг отключен' if job_removed else 'Постинг не был включен'
    update.message.reply_text(text)
    return BOT_STATE_INITIAL


def on_post_meme_once_pressed(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    meme_provider = context.dispatcher.user_data['meme_provider']
    img = meme_provider.get_next()
    if img is None:
        context.bot.send_message(chat_id, 'Не удалось получить мем: новые мемы в процессе генерации') 
    else:
        context.bot.send_photo(chat_id, img)
    return BOT_STATE_INITIAL


def on_enter_meme_starting_pressed(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Введите начало мема:')
    return BOT_STATE_TYPING_REPLY


def is_english(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def generate_meme_by_starting(update: Update, context: CallbackContext) -> None:
    text = update.message.text
    meme_provider = context.dispatcher.user_data['meme_provider']
    if len(text) > meme_provider.max_starting_text_length:
        update.message.reply_text('Ограничение на длину начального текста мема ' + str(meme_provider.max_starting_text_length))
    elif not is_english(text):
        update.message.reply_text('Допустим только английский текст')
    else:
        chat_id = update.message.chat_id
        img = meme_provider.get_starting_with(text.upper())
        if img is None:
            context.bot.send_message(chat_id, 'Сервер временно недоступен. Попробуйте позже') 
        else:
            context.bot.send_photo(chat_id, img)
    return BOT_STATE_INITIAL


def conversation_error(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Что-то пошло не так')
    return BOT_STATE_INITIAL  


def main():
    # Не недо использовать этот токен бота при тестировании:
    # легко и просто получить новый в телеграме у botfather
    updater = Updater('1388530746:AAHFjcytNgPwQwALnmWpIi-YNPn5S1VJ7Gc', use_context=True)
    dispatcher = updater.dispatcher
    with open('./memes_config.json', 'r') as config_file:
        memes_config = json.load(config_file)
    meme_provider = MemeProvider(memes_config)
    dispatcher.user_data['meme_provider'] = meme_provider
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            BOT_STATE_INITIAL: [
                MessageHandler(Filters.regex('^' + reply_keyboard[0][0] + '$'), on_post_meme_once_pressed),
                MessageHandler(Filters.regex('^' + reply_keyboard[1][0] + '$'), on_enter_meme_starting_pressed),
            ],
            BOT_STATE_TYPING_REPLY: [
                MessageHandler(Filters.text & ~(Filters.command), generate_meme_by_starting)
            ],
        },
        fallbacks=[MessageHandler(Filters.regex('^Done$'), conversation_error)],
        name="my_conversation"
    )
    dispatcher.add_handler(conv_handler)
    
    dispatcher.add_handler(CommandHandler('help', start))
    dispatcher.add_handler(CommandHandler('set', set_timer))
    dispatcher.add_handler(CommandHandler('unset', unset))
    
    updater.start_polling()

    # Block until you press Ctrl-C or the process receives SIGINT, SIGTERM or
    # SIGABRT. This should be used most of the time, since start_polling() is
    # non-blocking and will stop the bot gracefully.
    updater.idle()
    meme_provider.running = False

if __name__ == '__main__':
    main()