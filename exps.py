# Хисматуллин Владимир
#
# Модуль с экспериментами и графиками

import numpy as np
import torch
import torch.nn as nn

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from models import Language_Model, train, generate

def experiment(**kwargs):
    """
    kwargs: E - embedding_dim_list, H - hidden_dim_list, N - num_layers_list, D - dropout_list,
    V - vocab_length, train, test, loss_fn,  device, num_iter, silent_mode
    
    проводит серию экспериментов и возвращает список с списками качества для каждой модели.
    
    """
    n_models = len(kwargs['E'])
    results = [1] * n_models
    for i in range(n_models):
        model = Language_Model(kwargs['device'], vocab_length=kwargs['V'][i],
                                embedding_dim=kwargs['E'][i], hidden_dim=kwargs['H'][i], nlayers=kwargs['N'][i],
                                dropout=kwargs['D'][i]
                                ).to(kwargs['device'])
        optimizer = torch.optim.Adam(model.parameters(),
                             lr=kwargs['lr'])
        results[i] = train(
        kwargs['train'], kwargs['test'], model, kwargs['loss_fn'], optimizer, 
                kwargs['device'], kwargs['num_iter'], kwargs['silent_mode']
        )
    return results


def plot_results(res, n_models, title, model_strings, cpu_cuda_troubles=False):
    """
    Принимает результаты предыдущей функции, визуализирует их
    
    """
    plt.subplots(1, 2, figsize=(16,5))

    if cpu_cuda_troubles:
        cpu_res = []
        for tup in res:
            cpu_tup = []
            for one_res in tup:
                cur_res = [x.detach().cpu().item() for x in one_res]
                cpu_tup.append(cur_res)
            cpu_res.append(cpu_tup)
    else:
        cpu_res = res  

    ax = plt.subplot(1, 2, 1)
    ax.set_title('Зависимость функции потерь', fontsize=14)
    colors = ['C' + str(i) for i in range(n_models)]
    num_epoch = len(cpu_res[0][0])
    for i in range(n_models):
        sns.lineplot(x=np.arange(1, num_epoch + 1), y=cpu_res[i][0], color = colors[i], ax=ax)
    for i in range(n_models):
        sns.lineplot(x=np.arange(1, num_epoch + 1), y=cpu_res[i][2], color = colors[i], linestyle='--', ax=ax)
    ax.grid()
    ax.set_ylabel('Значение функции потерь', fontsize=14)
    ax.set_xlabel('Номер эпохи', fontsize=14)
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Зависимость accuracy', fontsize=14)
    for i in range(n_models):
        sns.lineplot(x=np.arange(1, num_epoch + 1), y=cpu_res[i][1], color = colors[i], ax=ax)
    for i in range(n_models):
        sns.lineplot(x=np.arange(1, num_epoch + 1), y=cpu_res[i][3], color = colors[i], linestyle='--', ax=ax)
    ax.grid()
    ax.set_ylabel('Значение accuracy', fontsize=14)
    ax.set_xlabel('Номер эпохи', fontsize=14)
    ax.legend([string + ', трейн' for string in model_strings] + [string + ', тест' for string in model_strings], loc='upper center', 
                 bbox_to_anchor=(-0.1, -0.15),fancybox=False, shadow=False, ncol=n_models,  fontsize=14)
    plt.suptitle(title, fontsize=14)
    plt.show()

def get_train_model(**kwargs):
    """
    Функция создания и обучения одной модели
    
    """
    model = Language_Model(kwargs['device'], vocab_length=kwargs['V'],
                          embedding_dim=kwargs['E'], hidden_dim=kwargs['H'], 
                          nlayers=kwargs['N'], dropout=kwargs['D']
                          ).to(kwargs['device'])
    optimizer = torch.optim.Adam(model.parameters(),
                            lr=kwargs['lr'])
    train(
      kwargs['train'], kwargs['test'], model, kwargs['loss_fn'], optimizer, 
              kwargs['device'], kwargs['num_iter'], kwargs['silent_mode']
    )
    return model

def experiment_gen (text_list, char_flag, length, encoder, decoder, model, device):
    """
    Генерирует и выводит текст для каждого заданного куска текста
    """
    for text in text_list:
        for k in [1, 2, 3, 5, 10, 15]:
            print('<------------ k = {:} ------------>'.format(k))
            if char_flag:
                print(''.join(text) + '   ----- генерация ----->')
                print((''.join(generate(model, text , 
                        encoder, decoder, length, {'method':'top-k', 'k':k}, device))))
            else:
                print(' '.join(text) + '   ----- генерация ----->')
                print((' '.join(generate(model, text , 
                        encoder, decoder, length, {'method':'top-k', 'k':k}, device))))