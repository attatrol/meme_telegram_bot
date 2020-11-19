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

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

"""
memes_config = [
    {
        'text_path': './disaster_girl/output.csv',
        'path_to_model': './disaster_girl/model',
        'template_path': './disaster_girl/template.jpg',
        'text_boxes': [
                (15, 15, 475, 115),
                (15, 270, 475, 370)
            ],
        'min_font_size': 30,
        'max_font_size': 70,
        'font_path': './UbuntuMono-B.ttf',
        'font_cell_ratio': 0.5
	}
]

"""
class Meme():
    def __init__(self, config):
        self.text_path = config['text_path']
        self.template_path = config['template_path']
        self.text_boxes = config['text_boxes']
        self.min_font_size = config['min_font_size']
        self.max_font_size = config['max_font_size']
        self.font_path = config['font_path']
        self.font_cell_ratio = config['font_cell_ratio']
        
        self.file = None

        self.try_start_reading()
        
    def try_start_reading(self):
        if not os.path.isfile(self.text_path):
            self.request_new_memes()
        
        self.file = open(self.text_path, 'r')
        
    def get_next(self):
        assert self.file is not None
        line = self.file.readline()
        if not line:
            self.file.close()
            self.request_new_memes()
            line = self.file.readline()
            assert line
        return line

    def request_new_memes(self):
        raise Exception('Not implemented')
        
# TODO вынести в конфиг
"""
Путь к шрифту специфичен для дистрибутива
"""

# резместить текст в середине box'а
def fit_text_to_box(text, text_box, font_size, font_cell_ratio):
    dx = text_box[2] - text_box[0]
    dy = text_box[3] - text_box[1]
    text_dx = ceil(font_size * font_cell_ratio * len(text))
    text_dy = font_size
    # assert dx >= text_dx and dy >= text_dy
    x = floor(text_box[0] + dx / 2 - text_dx / 2)
    y = floor(text_box[1] + dy / 2 - text_dy / 2)
    return (text, x, y, font_size)

# рекурсивная процедура для split_text
def split_text_ex(lengths, line_count, max_line_count, min_line_count):
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
    for i in range(1, len(lengths) - line_count):
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
            
# разбить текст на n макcимально близких по длине строк
def split_text(text, line_count):
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
        print(line_len)
        if max_len < line_len:
            max_len = line_len
        text_lines.append(text_line)
    print(text_lines)
    print(max_len)
    return text_lines, max_len

# вычислить многострочную верстку текста       
def get_text_layout(text, text_box, max_font_size, min_font_size, font_cell_ratio):
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
        return text.split('|')

    def add_text_to_template(self, path_to_img, texts, text_boxes, font_path, max_font_size, min_font_size, font_cell_ratio):
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