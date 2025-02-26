import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import random
import re

# Параметры
VOCAB_SIZE = 6000  # Увеличен для большего словарного запаса
HIDDEN_SIZE = 384  # Увеличен для лучшей обработки сложных зависимостей
EMBEDDING_DIM = 256  # Увеличен для лучшего представления слов
MAX_LENGTH = 64  # Увеличен для обработки более длинных предложений
EPOCHS = 1
BATCH_SIZE = 1
# TEMPERATURE = 0.8  # Параметр температуры для генерации  # Убрали, т.к. теперь передается в generate_responses

# Подготовка данных
class TextDataset(Dataset):
    def __init__(self, texts, vocab):
        self.texts = texts
        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        
    def __len__(self):
        return len(self.texts) - 1
    
    def __getitem__(self, idx):
        input_seq = self.texts[idx]
        target_seq = self.texts[idx + 1]
        
        input_tensor = self.seq_to_tensor(input_seq)
        target_tensor = self.seq_to_tensor(target_seq)
        return input_tensor, target_tensor
    
    def seq_to_tensor(self, seq):
        seq = seq[:MAX_LENGTH] if len(seq) > MAX_LENGTH else seq + ['<PAD>'] * (MAX_LENGTH - len(seq))
        tensor = torch.zeros(MAX_LENGTH, dtype=torch.long)
        for i, word in enumerate(seq):
            tensor[i] = self.word2idx.get(word, self.word2idx['<UNK>'])
        return tensor

# Улучшенная модель нейросети
class ChatRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.3):
        super(ChatRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)  # Добавляем механизм внимания
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        
        # Добавляем механизм внимания
        attention_weights = torch.softmax(self.attention(output), dim=1)
        output = output * attention_weights
        
        output = self.fc(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, hidden

# Функция для загрузки данных из текущей папки
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_texts = []
    for filename in os.listdir(current_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(current_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read().lower()
                # Разбиваем текст на предложения
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    words = sentence.strip().split()
                    if words:
                        all_texts.append(words)
    
    if not all_texts:
        print("В текущей папке нет .txt файлов для обучения.")
    return all_texts

# Создание словаря
def build_vocab(texts, vocab_size):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text)
    
    most_common = word_counts.most_common(vocab_size - 3)  # -3 для <PAD>, <UNK> и <EOS>
    vocab = ['<PAD>', '<UNK>', '<EOS>'] + [word for word, _ in most_common]
    return vocab

# Обучение модели с регуляризацией
def train_model(model, dataloader, epochs):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # игнорируем <PAD> токены
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # добавлена L2 регуляризация
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            output, _ = model(inputs)
            loss = criterion(output.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            # Градиентный клиппинг для предотвращения взрыва градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Эпоха {epoch+1}/{epochs}, Батч {batch_idx}/{len(dataloader)}, Потери: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)  # Корректировка скорости обучения
        print(f"Эпоха {epoch+1}/{epochs} завершена, Средние потери: {avg_loss:.4f}")

# Улучшенная генерация ответа
def generate_response(model, vocab, word2idx, idx2word, input_text, temperature):
    model.eval()
    words = input_text.lower().strip().split()
    input_words_set = set(words)  # Множество слов из входного текста

    input_seq = [word2idx.get(word, word2idx['<UNK>']) for word in words]
    input_seq = input_seq[:MAX_LENGTH] + [word2idx['<PAD>']] * (MAX_LENGTH - len(input_seq))
    input_tensor = torch.tensor([input_seq], dtype=torch.long)

    with torch.no_grad():
        output, hidden = model(input_tensor)

        # Используем sampling с температурой вместо argmax для разнообразия
        probs = torch.softmax(output[0, -1] / temperature, dim=0)

        # Уменьшаем вероятность слов из входного текста
        for word in input_words_set:
            if word in word2idx:
                probs[word2idx[word]] *= 0.1  # Снижаем вероятность повторения входных слов

        # Выбираем топ-5 слов с наибольшей вероятностью
        top_indices = torch.topk(probs, 5).indices.tolist()
        # Выбираем случайное слово из топ-5 с учетом их вероятностей
        next_word_idx = random.choices(top_indices, weights=[probs[i].item() for i in top_indices])[0]

        response = []
        current_word_idx = next_word_idx

        # Генерируем последовательность слов
        for _ in range(MAX_LENGTH):
            if current_word_idx == word2idx['<PAD>'] or current_word_idx == word2idx['<EOS>']:
                break

            word = idx2word[current_word_idx]
            response.append(word)

            # Подготовка для следующего шага
            next_input = torch.tensor([[current_word_idx]], dtype=torch.long)
            output, hidden = model(next_input, hidden)

            # Применяем температуру и sampling
            probs = torch.softmax(output[0, 0] / temperature, dim=0)

            # Уменьшаем вероятность повторения слов из ответа и входного текста
            for word in response + list(input_words_set):
                if word in word2idx:
                    probs[word2idx[word]] *= 0.1

            # Выбираем следующее слово
            if random.random() < 0.9:  # 90% времени выбираем из топ-5
                top_indices = torch.topk(probs, 5).indices.tolist()
                current_word_idx = random.choices(top_indices, weights=[probs[i].item() for i in top_indices])[0]
            else:  # 10% времени выбираем случайно для разнообразия
                current_word_idx = torch.multinomial(probs, 1).item()

        # Если ответ пустой или слишком короткий, генерируем заново
        if len(response) < 3:
            return "Извините, я не могу подобрать подходящий ответ."

        return ' '.join(response)

def generate_responses(model, vocab, word2idx, idx2word, input_text, temperatures=[0.5, 0.8, 1.0]):
    responses = {}
    for temp in temperatures:
        responses[temp] = generate_response(model, vocab, word2idx, idx2word, input_text, temp)
    return responses

# Главная функция
def main():
    # Загрузка данных
    print("Загрузка данных...")
    texts = load_data()
    if not texts:
        print("Нет данных для обучения. Завершение.")
        return
    
    # Создание словаря
    print("Создание словаря...")
    vocab = build_vocab(texts, VOCAB_SIZE)
    dataset = TextDataset(texts, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Инициализация модели
    print("Инициализация модели...")
    model = ChatRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE)
    
    # Обучение
    print("Начало обучения...")
    train_model(model, dataloader, EPOCHS)
    
    # Сохранение модели
    torch.save(model.state_dict(), 'chat_model.pth')
    print("Модель сохранена в 'chat_model.pth'")
    
    # Диалог
    word2idx = dataset.word2idx
    idx2word = dataset.idx2word
    print("\nОбучение завершено! Давай поговорим. Введи 'выход' для завершения.")
    
    # Сохраняем историю диалога для контекста
    conversation_history = []
    
    while True:
        user_input = input("Ты: ")
        if user_input.lower() == 'выход':
            break
            
        # Добавляем ввод пользователя в историю
        conversation_history.append(user_input)
        
        # Используем последние 3 реплики для контекста
        context = " ".join(conversation_history[-3:])
        # Генерируем несколько ответов с разными температурами
        responses = generate_responses(model, vocab, word2idx, idx2word, context)
        print("Бот:")
        for temp, response in responses.items():
            print(f"  (T={temp}): {response}")

        
        # Добавляем *один* из ответов бота в историю (например, с температурой 0.8)
        conversation_history.append(responses.get(0.8, "Извините, я не могу подобрать подходящий ответ."))

if __name__ == "__main__":
    main()