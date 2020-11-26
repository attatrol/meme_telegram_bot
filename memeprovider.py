#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:49:21 2020

@author: atta
"""
import threading
from random import randrange
from math import ceil, floor
import os
import os.path
import subprocess
import json
import pandas as pd
import time

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# при каком остатке в коллекции мемов начинать инференс
START_INFERENCE_DATA_OFFSET = 25
# максимальная длина пользовательского текста начала мема
USER_STARTING_TEXT_MAX_LENGTH = 50
# число попыток получить мем на один запрос
MAX_MEME_GET_TRIES = 3
# максимальное время на реквест мема, секунд
MEME_REQUEST_TIMEOUT = 18

class Meme():
    def __init__(self, config, owner):
        self.text_path = config['text_path']
        self.template_path = config['template_path']
        self.text_boxes = config['text_boxes']
        self.min_font_size = config['min_font_size']
        self.max_font_size = config['max_font_size']
        self.font_path = config['font_path']
        self.font_cell_ratio = config['font_cell_ratio']
        self.path_to_model = config['path_to_model']
        self.n_samples = config['n_samples']
        self.batch_size = config['batch_size']
        
        self.data = None
        self.data_index = 0
        self.data_size = -1
        self.try_start_reading()
        
        self.owner = owner
        self.start_inference_data_offset = START_INFERENCE_DATA_OFFSET
        self.inference_subprocess_token = None
        
    def try_start_reading(self):
        '''
        Начать чтение из файла
        '''
        if not os.path.isfile(self.text_path):
            self.request_new_memes()
        else:
            self.data = pd.read_csv(self.text_path, header = 0)
            self.data_index = 0
            self.data_size = len(self.data.index)

    def get_next(self):
        '''
        Вернуть следующий мем
        '''
        assert self.data is not None
        if self.data_index >= self.data_size and not self.request_new_memes(True):
            return None
        if self.data_index + self.start_inference_data_offset >= self.data_size:
            self.request_new_memes(False)
        result = self.data.iloc[self.data_index, 0]
        self.data_index += 1
        return result
    
    def get_user_text_meme_process(self, starting_text):
        '''
        Создать процесс генерации мема по пользовательскому начальному тексту
        '''
        expected_token = self.owner.subprocess_counter
        temp_params_dict_path = './temp_params_dict' + str(expected_token) + '.json'
        temp_output_path = './temp_output' + str(expected_token) + '.csv'
        path_to_params_dict = self.path_to_model + 'params_dict.json'
        with open(path_to_params_dict, 'r') as f:
            params_dict = json.load(f)
        params_dict['prefix'] = params_dict['prefix'] + starting_text
        with open(temp_params_dict_path, 'w') as f:
            json.dump(params_dict, f)
        cmd = ['python', 
               'inference.py',
               '--path-to-model-dir=' + self.path_to_model,
               '--path-to-params-dict=' + temp_params_dict_path,
               '--output-path=' + temp_output_path,
               '--n-samples=1',
               '--batch-size=1'
               ]
        token = self.owner.queue_process(cmd)
        assert expected_token == token
        return token, temp_params_dict_path, temp_output_path
        
    def get_user_text_meme(self, token, temp_params_dict_path, temp_output_path):
        '''
        Получить пользовательский мем
        '''
        if not self.owner.process_finished(token):
            return None
        data = pd.read_csv(temp_output_path, header = 0)
        os.remove(temp_params_dict_path)
        os.remove(temp_output_path)
        return data.iloc[0, 0]

    def request_new_memes(self, loadData: bool) -> bool:
        '''
        Запросить у модели следующий батч мемов
        '''
        if self.inference_subprocess_token is None:
            cmd = ['python', 
                   'inference.py',
                   '--path-to-model-dir=' + self.path_to_model,
                   '--path-to-params-dict=' + self.path_to_model + 'params_dict.json',
                   '--output-path=' + self.text_path,
                   '--n-samples=' + str(self.n_samples),
                   '--batch-size=' + str(self.batch_size)
                   ]
            self.inference_subprocess_token = self.owner.queue_process(cmd)
            return False
        elif not self.owner.process_finished(self.inference_subprocess_token):
            return False
        else:
            if loadData:
                self.inference_subprocess_token = None
                self.data = pd.read_csv(self.text_path, header = 0)
                self.data_index = 0
                self.data_size = len(self.data.index)
            return True


def fit_text_to_box(text, text_box, font_size, font_cell_ratio):
    '''
    Разместить текст в середине box'а
    '''
    dx = text_box[2] - text_box[0]
    dy = text_box[3] - text_box[1]
    text_dx = ceil(font_size * font_cell_ratio * len(text))
    text_dy = font_size
    # assert dx >= text_dx and dy >= text_dy
    x = floor(text_box[0] + dx / 2 - text_dx / 2)
    y = floor(text_box[1] + dy / 2 - text_dy / 2)
    return (text, x, y, font_size)

def split_text_ex(lengths, line_count, max_line_count, min_line_count):
    '''
    Рекурсивная процедура для split_text
    '''
    assert line_count > 0
    results = []
    if len(lengths) == 0:
        return None
    if line_count == 1:
        length_sum = len(lengths) - 1
        for length in lengths:
            length_sum += length[1]
        if length_sum > max_line_count:
            max_line_count = length_sum
        if length_sum < min_line_count:
            min_line_count = length_sum
        return ([lengths], (max_line_count, min_line_count))
    for i in range(1, len(lengths) - line_count + 2):
        first_line_len = i - 1 # пробелы
        for j in range(i):
            first_line_len +=lengths[j][1]
        other_words = lengths[i : ]
        result = split_text_ex(other_words, line_count - 1, max_line_count, min_line_count)
        if result is None:
            continue
        first_line_words = lengths[ : i]
        if result[1][0] < first_line_len:
            result = (result[0], (first_line_len, result[1][1]))
        if result[1][1] > first_line_len:
            result = (result[0], (result[1][0], first_line_len))
        result[0].insert(0, first_line_words)
        results.append(result)
    if len(results) == 0:
        return None
    best_index = -1
    best_max_length = float("inf")
    best_min_length = 0
    for i in range(0, len(results)):
        if results[i] is not None:
            max_length = results[i][1][0]
            min_length = results[i][1][1]
        if  max_length < best_max_length or (max_length == best_max_length and min_length > best_min_length):
            best_max_length = max_length
            best_min_length = min_length
            best_index = i
    return None if best_index == -1 else results[best_index]                   
            

def split_text(text, line_count):
    '''
    Разбить текст на n макcимально близких по длине строк
    '''
    words = text.split(' ')
    lengths = []
    i = 0
    for word in words:
        lengths.append((i, len(word)))
        i += 1
    optimal_length = ceil(len(text) / line_count)
    split_result = split_text_ex(lengths, line_count, optimal_length, optimal_length)
    if split_result is None:
        return None, 0
    max_len = 0
    text_lines = []
    for line in split_result[0]:
        line_len = len(line) - 1 + line[0][1]
        text_line = words[line[0][0]]
        for i in range(1, len(line)):
            text_line += ' ' + words[line[i][0]]
            line_len += line[i][1]
        if max_len < line_len:
            max_len = line_len
        text_lines.append(text_line)
    return text_lines, max_len

     
def get_text_layout(text, text_box, max_font_size, min_font_size, font_cell_ratio):
    '''
    Вычислить многострочную верстку текста
    '''
    dx = text_box[2] - text_box[0]
    dy = text_box[3] - text_box[1]
    assert dx > 0 and dy > 0
    if dy < min_font_size:
        return False, []
    max_font_size = dy if max_font_size > dy else max_font_size
    one_line_font_size = floor(dx / (len(text) * font_cell_ratio))
    # типичный случай, однострочный текст:
    if one_line_font_size > min_font_size:
        if one_line_font_size > max_font_size:
            one_line_font_size = max_font_size
        return True, [fit_text_to_box(text, text_box, one_line_font_size, font_cell_ratio)]
    else:
        max_lines_in_box = floor(dy / min_font_size)
        for line_count in range(2, max_lines_in_box):
            split, max_length = split_text(text, line_count)
            if split is None:
                return False, []
            font_size = floor(dx / (max_length * font_cell_ratio))
            if font_size > min_font_size:
                if font_size > max_font_size:
                    font_size = max_font_size
                result = []
                for i in range(line_count):
                    split_text_box = (text_box[0], floor(text_box[1] + i * dy / line_count), text_box[2], ceil(text_box[1] + (i + 1) * dy / line_count))
                    result.append(fit_text_to_box(split[i], split_text_box, font_size, font_cell_ratio))
                return True, result
    return False, []
    
class MemeProvider():
    def __init__(self, config):
        self.lock = threading.Lock()
        self.memes = []
        for meme_config in config:
            self.memes.append(Meme(meme_config, self))
        assert len(self.memes) > 0
        
        self.max_starting_text_length = USER_STARTING_TEXT_MAX_LENGTH

        self.subprocess_counter = 0
        self.subprocesses = {}
        self.subprocesses_queue = []
        
        self.running = True
        self.run_subprocesses()
        
    def split_into_boxes(self, text):
        '''
        Разбить текст мема на фрагменты для каждого box'а
        '''
        return text.split('|')

    def add_text_to_template(self, path_to_img, texts, text_boxes, font_path, max_font_size, min_font_size, font_cell_ratio):
        '''
        Сгенерировать изображение из шаблона и текста
        '''
        if len(texts) > len(text_boxes):
            return None
        img = Image.open(path_to_img)
        draw = ImageDraw.Draw(img)
        for i in range(len(texts)):
            success, layout = get_text_layout(texts[i], text_boxes[i], max_font_size, min_font_size, font_cell_ratio)
            if not success:
                return None
            # (text, x, y, font_size)
            for text_drawing in layout:
                font = ImageFont.truetype(font_path, text_drawing[3])
                shadowcolor = (0, 0, 0)
                fillcolor = (255,255,255)
                draw.text((text_drawing[1] - 1, text_drawing[2] - 1), text_drawing[0], font = font, fill = shadowcolor)
                draw.text((text_drawing[1] + 1, text_drawing[2] - 1), text_drawing[0], font = font, fill = shadowcolor)
                draw.text((text_drawing[1] - 1, text_drawing[2] + 1), text_drawing[0], font = font, fill = shadowcolor)
                draw.text((text_drawing[1] + 1, text_drawing[2] + 1), text_drawing[0], font = font, fill = shadowcolor)
                draw.text((text_drawing[1], text_drawing[2]), text_drawing[0], font = font, fill = fillcolor)
        img.save('./temp.jpg')
        return open('./temp.jpg', 'rb')
    
    def run_subprocesses(self):
        owner = self
        def run_in_thread(owner):
            token = None
            while owner.running:
                with owner.lock:
                    if len(owner.subprocesses_queue) > 0:
                        token = owner.subprocesses_queue.pop(0)
                    else:
                        token = None
                if token is None:
                    time.sleep(0.05)
                else:
                    with owner.lock:
                        proc = subprocess.Popen(owner.subprocesses[token][1])
                        owner.subprocesses[token] = (1, proc)
                    proc.wait()
                    with owner.lock:
                        owner.subprocesses[token] = (2, None)
            return
        thread = threading.Thread(target=run_in_thread, args=(owner,))
        thread.start()
        return
    
    def queue_process(self, cmd):
        '''
        Зарегистрировать процесс.
        Принимает параметры процесса.
        Возвращает токен процесса.
        '''
        with self.lock:
            token = self.subprocess_counter
            self.subprocess_counter += 1
            self.subprocesses_queue.append(token)
            self.subprocesses[token] = (0, cmd)
            return token


    def cancel_process(self, token) -> bool:
        '''
        Отменить выполнение процесса в очереди
        '''
        with self.lock:
            for i in len(self.subprocesses_queue):
                if self.subprocesses_queue[i] == token:
                    self.subprocesses_queue.pop(token)
                    self.subprocesses[token] = (2, None)
                    return True
            return False
            
    def process_finished(self, token):
        '''
        Проверить, что процесс завершен
        '''
        with self.lock:
            return token not in self.subprocesses or self.subprocesses[token][0] == 2
        
    
    def get_next(self):
        '''
        Получить следующий мем
        '''
        start_time = time.time()
        while MEME_REQUEST_TIMEOUT > time.time() - start_time:
            meme = self.memes[randrange(len(self.memes))]
            text = meme.get_next()
            if text is not None:
                text = self.split_into_boxes(text)
            if text is not None:
                img = self.add_text_to_template(meme.template_path, text, meme.text_boxes, meme.font_path, \
                                                meme.max_font_size, meme.min_font_size, meme.font_cell_ratio)
                if img is not None:
                    return img
            time.sleep(0.1)
        return None


    def get_starting_with(self, starting_text):
        '''
        Получить один мем на основе начала его текста
        '''
        start_time = time.time()
        meme = self.memes[randrange(len(self.memes))]
        token, temp_params_dict_path, temp_output_path = meme.get_user_text_meme_process(starting_text)
        while MEME_REQUEST_TIMEOUT > time.time() - start_time:
            text = meme.get_user_text_meme(token, temp_params_dict_path, temp_output_path)
            if text is not None:
                text = self.split_into_boxes(text)
                return self.add_text_to_template(meme.template_path, text, meme.text_boxes, meme.font_path, \
                                            meme.max_font_size, meme.min_font_size, meme.font_cell_ratio)
        self.cancel_process(token)        
        return None
