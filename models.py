# Хисматуллин Владимир
#
# Модуль с языковой моделью на основе LSTM и функциями обучения

import numpy as np
import torch
import torch.nn as nn

class Language_Model(torch.nn.Module):
    def __init__(
        self, device, vocab_length, embedding_dim, hidden_dim, nlayers,
        rec_layer=torch.nn.LSTM, dropout=None, **kwargs
        ):
        """
        vocab_length - количество слов в алфавите
        embedding_dim - размерность эмбеддингов
        hidden_dim - размерность скрытого слоя
        nlayers - если > 1, то LSTM будет многослойной
        dropout - возможный dropout
        
        """
        super().__init__()

        self.device = device
        self.dropout = dropout
        
        self.vocab_length = vocab_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Эмбеддинги
        self.word_embeddings = torch.nn.Embedding(vocab_length, embedding_dim)
        
        #  Слой LSTM 
        if dropout is not None:
            self.rnn = rec_layer(self.embedding_dim, self.hidden_dim, num_layers=nlayers,
                                 dropout=self.dropout, **kwargs)
        else:
            self.rnn = rec_layer(self.embedding_dim, self.hidden_dim, num_layers=nlayers,
                                 **kwargs)
        
        #  Слой для предсказания вероятностей
        self.output = torch.nn.Linear(self.hidden_dim, self.vocab_length)

    
    def forward(self, tokens):
        """
        На входе токены 
        
        На выходе предсказания следующими за ними
        
        """
        # L - длина предложений, B - batch_size, E - embedding_dim, H -hidden_dim
        # На входе tokens.shape == (L, B)
        x = self.word_embeddings(tokens) #x.shape == (L, B, E)
        
        # Слой LSTM
        output_lstm, (hidden, cell) = self.rnn(x) # output_lstm.shape == (L, B, H)
        
        
        output_lstm = output_lstm.view(-1, self.hidden_dim) # output_lstm.shape == (L * B, H)
        output = self.output(output_lstm) # output.shape == (L * B, vocab_length)
        
        return output.view(-1, tokens.shape[1], self.vocab_length)
        

    def generate(self, seq, n_words, mode): 
        """
        seq - входная последовательность токенов размерности L
        mode - словарь - способ генерации 
        {'method': 'greedy' / 'top-k' / 'beam' - методы генерации текста
         'k': параметр k для top-k}

        На выходе предсказанное предложение длины n_words из токенов
        
        Код генерации частично заимствован из нашей практической работы
        
        """
        generated_words = []
        
        # Делаем первое предсказание
        embeddings = self.word_embeddings(seq).unsqueeze(1) # L x 1 x E
        output_lstm, hidden = self.rnn(embeddings) # L x 1 x  H
        output = output_lstm[-1] # 1 x H
        scores = self.output(output) # 1 x V

        # В цикле генерируем токены
        for i in range(n_words):
            if mode['method'] == 'greedy':
                # жадный поиск для генерации слов
                _, current_word = torch.max(scores, dim=1) # 1 x 1
                
            elif mode['method']  == 'top-k':
                # Семплирование из top-k
                top_values, top_indices = torch.topk(scores.squeeze(), mode['k'])
                p = torch.nn.functional.softmax(top_values, dim=0).cpu().detach().numpy()
                current_word = torch.Tensor.int(torch.Tensor([
                    np.random.choice(top_indices.cpu().detach().numpy(), p=p)])).to(self.device)
            
            # Добавляем слово к сгенерированным
            generated_words.append(current_word)

            # Делаем новое предсказание
            embeddings = self.word_embeddings(current_word).unsqueeze(0) # 1 x 1 x E
            output_lstm, hidden = self.rnn(embeddings, hidden) # 1 x 1 x H
            output = output_lstm[0] # 1 x H 
            scores = self.output(output) # V
        
        return torch.cat(generated_words, dim=0)


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    """
    Обучаем одну эпоху
    
    """
    model.train()
    for idx, data in enumerate(dataloader):

        # 1. Take data from batch
        tokens, targets = data['objects'], data['targets']
        tokens, targets = tokens.to(device), targets.to(device)

        # 2. Perform forward pass
        preds = model(tokens)

        # 3. Evaluate loss
        loss = loss_fn(preds.view(-1, preds.size(2)), targets.view(-1)) 

        # 4. Make optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
          
def evaluate(dataloader, model, loss_fn, device):
    """
    Оцениваем качество сети на dataloader
    
    """
    model.eval()
    
    total_size = 0
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            # 1. Take data from batch
            tokens, targets = data['objects'], data['targets']
            tokens, targets = tokens.to(device), targets.to(device)
            
            # один раз в конце делить acc - не совсем верно, батчи могут иметь немного разный размер
            total_size += targets.shape[0] * targets.shape[1]

            # 2. Perform forward pass
            preds = model(tokens) 
            
            # 3. Evaluate loss
            total_loss += (loss_fn(preds.view(-1, preds.size(2)), targets.view(-1)) *
                           targets.shape[0] * targets.shape[1])
            

            # 4. Evaluate accuracy
            total_accuracy += (preds.argmax(2) == targets).sum()
              
    return total_loss / total_size, total_accuracy / total_size
    

def train(
    train_loader, test_loader, model, loss_fn, optimizer, device, num_epochs, silent_mode=True
):
    """
    Обучаем сеть num_epochs эпох, считаем качество

    Возвращаем loss, accuracy на трейне, затем на тесте
    
    """
    test_losses = []
    train_losses = []
    test_accuracies = []
    train_accuracies = []
    for epoch in range(num_epochs):
        # Эпоха
        train_epoch(train_loader, model, loss_fn, optimizer, device)
        
        # Оцениваем на train
        train_loss, train_acc = evaluate(train_loader, model, loss_fn, device)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        
        # Оцениваем на тесте
        test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        if not silent_mode:
            print(
                'Epoch: {0:d}/{1:d}. Loss (Train/Test): {2:.3f}/{3:.3f}. Accuracy (Train/Test): {4:.3f}/{5:.3f}'.format(
                    epoch + 1, num_epochs, train_losses[-1], test_losses[-1], train_accuracies[-1], test_accuracies[-1]
                )
            )
    return train_losses, train_accuracies, test_losses, test_accuracies

def encode_sentance(sentence, encoder):
    """
    Переводит предложение в список токены
    sentence - список слов
    encoder - словарь: отображает слова в токены
    
    """
    
    return np.array([encoder[word] for word in sentence], dtype=np.int64)


def decode_sentance(sentence, decoder):
    """
    Переводит список токенов в предложение
    sentence - список слов
    decoder - словарь: отображает токены в слова 
    
    """
    return np.array([decoder[word] for word in sentence])


def generate(model, sentence, encoder, decoder, nwords, mode, device):
    """
    Принимает на вход строку из слов и генерирует текст
    
    sentence - строка со словами
    encoder - кодировщик ( слова -> токены )
    decoder - декодировщик ( токены -> слова )
    nwords - количество генерируемых слов
    
    """
    seq = encode_sentance(sentence, encoder)
    seq = torch.tensor(seq).to(device)
    out = model.generate(seq, nwords, mode=mode)
    return decode_sentance(out.cpu().detach().numpy(), decoder)