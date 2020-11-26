#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:49:21 2020

@author: atta
"""
import threading
from random import randrange
from math import ceil, floor
import os.path
import subprocess
import json
import pandas as pd

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class Meme():
    def __init__(self, config):
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
        if self.data_index >= self.data_size:
            self.request_new_memes()
        result = self.data.iloc[self.data_index, 0]
        self.data_index += 1
        return result
    
    def get_starting_with(self, starting_text):
        '''
        Вернуть следующий мем
        '''
        path_to_params_dict = self.path_to_model + 'params_dict.json'
        with open(path_to_params_dict, 'r') as f:
            params_dict = json.load(f)
        params_dict['prefix'] = params_dict['prefix'] + starting_text
        with open('./temp_params_dict.json', 'w') as f:
            json.dump(params_dict, f)
        cmd = ['python', 
               'inference.py',
               '--path-to-model-dir=' + self.path_to_model,
               '--path-to-params-dict=./temp_params_dict.json',
               '--output-path=./temp_output.csv',
               '--n-samples=1',
               '--batch-size=1'
               ]
        subprocess.Popen(cmd).wait() 
        data = pd.read_csv('./temp_output.csv', header = 0)
        return data.iloc[0, 0]

    def request_new_memes(self):
        '''
        Запросить у модели следующий батч мемов
        '''
        cmd = ['python', 
               'inference.py',
               '--path-to-model-dir=' + self.path_to_model,
               '--path-to-params-dict=' + self.path_to_model + 'params_dict.json',
               '--output-path=' + self.text_path,
               '--n-samples=' + str(self.n_samples),
               '--batch-size=' + str(self.batch_size)
               ]
        subprocess.Popen(cmd).wait() 
        self.data = pd.read_csv(self.text_path, header = 0)
        self.data_index = 0
        self.data_size = len(self.data.index)

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
            self.memes.append(Meme(meme_config))
        assert len(self.memes) > 0
        
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
    
    def get_next(self):
        '''
        Получить следующий мем
        '''
        with self.lock:
            meme = self.memes[randrange(len(self.memes))]
            text = self.split_into_boxes(meme.get_next())
            img = self.add_text_to_template(meme.template_path, text, meme.text_boxes, meme.font_path, \
                                            meme.max_font_size, meme.min_font_size, meme.font_cell_ratio)
            while not img:
                text = self.split_into_boxes(meme.get_next())
                img = self.add_text_to_template(meme.template_path, text, meme.text_boxes, meme.font_path, \
                                                meme.max_font_size, meme.min_font_size, meme.font_cell_ratio)
            return img
        
    def get_starting_with(self, starting_text):
        '''
        Получить один мем на основе начала его текста
        '''
        with self.lock:
            meme = self.memes[randrange(len(self.memes))]
            text = self.split_into_boxes(meme.get_starting_with(starting_text))
            img = self.add_text_to_template(meme.template_path, text, meme.text_boxes, meme.font_path, \
                                            meme.max_font_size, meme.min_font_size, meme.font_cell_ratio)
            while not img:
                text = self.split_into_boxes(meme.get_starting_with(starting_text))
                img = self.add_text_to_template(meme.template_path, text, meme.text_boxes, meme.font_path, \
                                                meme.max_font_size, meme.min_font_size, meme.font_cell_ratio)
            return img