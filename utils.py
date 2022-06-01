# Хисматуллин Владимир
#
# Модуль с обработкой текста и словарями

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
from functools import partial
from collections import defaultdict

import regex
import re


def tokenize(text, language, charsplit=False, 
             words_to_delete=None, stopwords=None, stemmer=None):
    """
    Токенизация текста:

    words_to_delete - список строк, подлежащих удалению
    language - язык
    text - строка
    charsplit - текст разбивается по символам, иначе по словам
    stopwords, stemmer - если указаны, то используются при обработке предложения
    
    """
    if words_to_delete is not None:
        for string in words_to_delete:
            text = re.sub(string, '', text)

    text = text.lower()
    if not charsplit:
        if language == 'ru':
            text = re.sub('[^а-я.,]+', ' ', text)
        else:
            text = re.sub('[^a-z.,]+', ' ', text)

        text = re.sub("[.]+", " .", text)
        text = re.sub("[,]+", " ,", text)

        # Обрабатываем stemmer'ы и stopwords
        if stemmer is not None:
            if stopwords is not None:
                text = [stemmer.stem(word) for word in re.split(r' +', text) 
                        if (word not in stopwords) and (word)]
            else:
                text = [stemmer.stem(word) for word in re.split(r' +', text) if (word) ]
        elif stopwords is not None:
            text = [word for word in re.split(r' +', text)  
                    if (word not in stopwords)  and (word) ]
        else:
            text = [word for word in re.split(r' +', text)  if (word) ]
    else:
        # По буквам тут отличие в обработке, нужно схлопнуть пробелы
        if language == 'ru':
            text = re.sub('[^а-я.,!]+', ' ', text)
        else:
            text = re.sub('[^a-z.,!]+', ' ', text)
        text = re.sub(" +", " ", text) 
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" !", "!")
      
    return text

def get_text_lines (filename, language, charsplit=False, 
                    words_to_delete=None, stopwords=None, stemmer=None):
    """
    Считывает файл и токинизирует
    Возвращает текст

    Параметры - параметры токенизациии
    
    """
    with open(filename, 'r', encoding="utf8") as f:
        text_lines = f.readlines()
    text = tokenize(' '.join(text_lines), language, charsplit, 
                    words_to_delete, stopwords, stemmer)
    return text

def create_encoders(text, min_frequency=10):
    """
    На входе: 
    min_frequency - минимальная частота, все слова с меньшей удаляются

    На выходе текст без непопулярных слов 
        + декодер, кодировщик
        ( кодировщик - отражение слов в токены
          декодер - отражение токенов в слова  )
    
    """
    # Считаем слова
    counter = defaultdict(int)
    for token in text:
        counter[token] += 1
     
    # Сортируем словарь
    values = np.array(list(counter.values()))
    keys = np.array(list(counter.keys()))
    sorted_indeces = np.argsort(values)[::-1]
    pruned_dict = [(key, value) for key, value in \
                   zip(keys[sorted_indeces], values[sorted_indeces]) if value >= min_frequency]
    rare_words = [key for key, value in \
                   zip(keys[sorted_indeces], values[sorted_indeces]) if value < min_frequency]
    
    text = [word for word in text if word not in rare_words]
    ordered_dict = dict(pruned_dict)
    
    # Создаём отображение слов в токены
    my_vocab = {}
    decoding_vocab = {}
    for i, key in enumerate(ordered_dict.keys()):
        my_vocab[key] = i
        decoding_vocab[i] = key
    
    return text, my_vocab, decoding_vocab


class BookDataset(Dataset):
    def __init__(self, text_words, vocab, max_len, device, intersection=0):
        """
        text_words - список слов - слова из выборки
        vocab - словарь, по которому можно отобразить слова в токены
        
        max_len - длина предложения
        
        intersection - длина наложения соседних предложений
        ...    Например дан текст: шла Саша по Шоссе и сосала сушку
        ...    max_len = 3, intersection = 2 :
        ...        шла Саша по, Саша по Шоссе, по Шоссе и, Шоссе и сосала, и сосала сушку
        ...    max_len = 3, intersection = 1:
        ...        шла Саша по, по Шоссе и, и сосала сушку
        ...    max_len = 3, intersection = 0:
        ...        шла Саша по, Шоссе и сосала 

        """
        super().__init__()
        
        self.device = device
        self.max_len = max_len
        self.intersection = intersection
        self.num_sections = len(text_words) // (self.max_len - self.intersection)
        self.text = text_words[:(self.max_len - self.intersection) * self.num_sections]
        self.tokens = [vocab[x] for x in text_words]
        
        
    def __getitem__(self, idx):
        """
        Принимает индекс объекта.
        Возвращает словарь следующего вида:
            {
                'text' str: текст предложения,
                'tokens' dict: 'object': предложение в токенах, кроме последнего слова
                               'target': предложение в токенах, кроме первого слова
            }
        
        """
        #token_data = [-1] + self.tokens[idx * (self.max_len - self.intersection): 
        token_data = self.tokens[idx * (self.max_len - self.intersection): 
                           idx * (self.max_len - self.intersection) + self.max_len]
        text_data = self.text[idx * (self.max_len - self.intersection): 
                              idx * (self.max_len - self.intersection) + self.max_len]
        return {'tokens': {'object': torch.Tensor.long(torch.Tensor(token_data[:-1]).to(self.device)),
                          'target': torch.Tensor.long(torch.Tensor(token_data[1: ]).to(self.device))},
                'text': text_data[:-1]}
    
    def __len__(self):
        """
        Возвращает кол-во датасетов
        
        """
        return self.num_sections

def collate_fn(batch):
    """
    Батч размера B x L отображает в L x B
    
    """
    ret_dict = {}
    ret_dict['text'] = [dict['text'] for dict in batch]
    
    objects = [dict['tokens']['object'] for dict in batch]
    targets =  [dict['tokens']['target'] for dict in batch]

    objects = torch.nn.utils.rnn.pad_sequence(objects,
                                    batch_first=False, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets,
                                    batch_first=False, padding_value=0)

    ret_dict['objects'] = objects
    ret_dict['targets'] = targets
    
    return ret_dict


def get_data_loaders (text, encoder, max_len, intersection, device, test_size=0.3, shuffle=True, batch_size=100):
    """
    text, encoder, max_len, intersection, device - параметры в  BookDataset
    test_size - процентный размер тестовой выборки
    shuffle, batch_size - параметры в DataLoader
    
    Возвращает: 
        train_data_loader, test_data_loader
    
    """

    # Создаём  BookDataset'ы
    length = len(text)
    train_length = int((1 - test_size) * length)
    train_dataset = BookDataset(text[:train_length], encoder, max_len, device, intersection=intersection)
    test_dataset = BookDataset(text[train_length:], encoder, max_len, device, intersection=intersection)
    
    # Создаём Dataloader'ы
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, collate_fn = collate_fn)
    test_loader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, collate_fn = collate_fn, drop_last=True)
    
    return train_loader, test_loader