import torch  # Добавляем импорт torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара
import torch.nn.functional as F  # Добавляем импорт F для использования softmax
import onnx
import onnxruntime as ort

# Устанавливаем устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextDataset(Dataset):
    def __init__(self, texts, seq_length):
        self.seq_length = seq_length
        self.text = ''.join(texts)
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = self.prepare_data()

    def prepare_data(self):
        data = []
        for i in range(0, len(self.text) - self.seq_length):
            seq_in = self.text[i:i + self.seq_length]
            seq_out = self.text[i + self.seq_length]
            data.append((seq_in, seq_out))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_in, seq_out = self.data[idx]
        x = torch.tensor([self.char_to_idx[ch] for ch in seq_in], dtype=torch.long)
        y = torch.tensor(self.char_to_idx[seq_out], dtype=torch.long)
        return x, y

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.3, rnn_type='LSTM', bidirectional=False):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                hidden_size, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                hidden_size, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )
        else:
            raise ValueError("Неподдерживаемый тип RNN. Используйте 'LSTM' или 'GRU'.")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.rnn(x, hidden)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device),
                    torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device))
        else:
            return torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

class ChatHistory:
    def __init__(self):
        self.history = []

    def add_message(self, message):
        self.history.append(message)

    def get_context(self, max_length):
        context = ' '.join(self.history[-max_length:])
        return context

def train_model(model, dataset, num_epochs, batch_size, seq_length, lr, model_dir='models'):
    if len(dataset) == 0:
        raise ValueError("Датасет пуст. Убедитесь, что данные загружены корректно.")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=num_epochs)

    model.to(device)
    # Убираем вызов несуществующего метода gradient_checkpointing_enable
    # model.gradient_checkpointing_enable()  # Включаем Gradient Checkpointing
    # Убираем компиляцию модели для ускорения инференса
    # model = torch.compile(model)  # Компилируем модель для ускорения инференса

    scaler = torch.amp.GradScaler('cuda')  # Используем автоматическое смешанное обучение

    for epoch in range(num_epochs):
        hidden = model.init_hidden(batch_size)
        total_loss = 0
        # Используем tqdm для отображения прогресса
        for x, y in tqdm(dataloader, desc=f'Эпоха {epoch + 1}/{num_epochs}'):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            hidden = tuple([h.detach() for h in hidden])
            if x.size(0) != batch_size:
                hidden = model.init_hidden(x.size(0))
            with torch.amp.autocast('cuda'):  # Используем автоматическое смешанное обучение
                output, hidden = model(x, hidden)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        print(f'Средняя потеря: {avg_loss:.4f}')
    
    # Сохраняем модель после завершения всех эпох
    version = f"{num_epochs // 10}.{num_epochs % 10}.{0}"
    model_path = os.path.join(model_dir, f'model_v{version}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Модель сохранена в {model_path}')

    # Экспортируем модель в ONNX
    dummy_input = torch.randint(0, len(dataset.chars), (1, seq_length)).to(device)
    torch.onnx.export(model, (dummy_input, model.init_hidden(1)), os.path.join(model_dir, f'model_v{version}.onnx'), opset_version=11)
    print(f'Модель экспортирована в ONNX формат в {os.path.join(model_dir, f"model_v{version}.onnx")}')

def generate_text(model, dataset, start_str, length, temperature=0.7):
    model.eval()
    hidden = model.init_hidden(1)
    try:
        input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in start_str], dtype=torch.long).unsqueeze(0).to(device)
    except KeyError:
        print("Ошибка: В начальной строке содержатся символы, которых нет в обучающих данных")
        return ""
    
    generated_text = start_str

    with torch.no_grad():  # Добавляем для оптимизации
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            
            # Применяем температуру для увеличения разнообразия
            output = output.div(temperature)
            
            # Используем вероятностное сэмплирование
            probs = F.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probs[0], 1)
            
            next_char = dataset.idx_to_char[next_char_idx.item()]
            generated_text += next_char
            input_seq = next_char_idx.unsqueeze(0)

    return generated_text

def log_conversation(user_input, response):
    with open("conversation_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"User: {user_input}\n")
        log_file.write(f"AI: {response}\n\n")

def save_user_input(user_input):
    with open("data/user_inputs.txt", "a", encoding="utf-8") as user_file:
        user_file.write(f"{user_input}\n")

if __name__ == "__main__":
    seq_length = 100  # Уменьшаем длину последовательности для повышения производительности
    hidden_size = 256  # Уменьшаем размер скрытого слоя для повышения производительности
    num_layers = 2    # Уменьшаем количество слоев для повышения производительности
    num_epochs = 1   # Уменьшаем количество эпох для повышения производительности
    batch_size = 4    # Меньше батч для лучшей генерализации
    lr = 0.001        # Уменьшаем learning rate
    rnn_type = 'LSTM' # Тип RNN
    bidirectional = False # Двунаправленный RNN

    texts = []
    for filename in os.listdir('data'):
        if filename.endswith('.txt'):
            with open(os.path.join('data', filename), "r", encoding="utf-8") as file:
                texts.append(file.read())

    dataset = TextDataset(texts, seq_length)
    model = CharRNN(len(dataset.chars), hidden_size, num_layers, rnn_type=rnn_type, bidirectional=bidirectional)
    
    # Загружаем модель, если она существует
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.startswith('model_v') and f.endswith('.pth')]
    if model_files:
        latest_model = sorted(model_files, key=lambda x: list(map(int, x.split('_v')[1].split('.pth')[0].split('.'))))[-1]
        load_model = input(f"Найдено {len(model_files)} версий модели. Загрузить последнюю версию {latest_model}? (y/n): ")
        if load_model.lower() == 'y':
            try:
                model.load_state_dict(torch.load(os.path.join(model_dir, latest_model), map_location=device))
                print(f'Модель загружена из {latest_model}')
            except Exception as e:
                print(f"Ошибка при загрузке модели: {e}")
    
    train_model(model, dataset, num_epochs, batch_size, seq_length, lr, model_dir)

    chat_history = ChatHistory()

    while True:
        user_input = input("Введите начальную строку: ")
        if not user_input:
            break
        save_user_input(user_input)
        chat_history.add_message(user_input)
        context = chat_history.get_context(seq_length)
        # Пробуйте разные температуры
        for temp in [0.7, 0.8, 0.9]:
            print(f"\nТемпература {temp}:")
            generated_text = generate_text(model, dataset, context, 200, temperature=temp)
            print(generated_text)
            log_conversation(user_input, generated_text)
            chat_history.add_message(generated_text)