import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ai import TextDataset  # Импортируем TextDataset из ai.py
import os

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.5):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Добавляем dropout для предотвращения переобучения
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.rnn(x, hidden)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

def generate_text(model, dataset, start_str, length, temperature=0.8):
    model.eval()
    hidden = model.init_hidden(1)
    try:
        input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in start_str], 
                               dtype=torch.long).unsqueeze(0)
    except KeyError:
        print("Ошибка: В начальной строке содержатся символы, которых нет в обучающих данных")
        return ""
    
    generated_text = start_str

    with torch.no_grad():  # Отключаем вычисление градиентов
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            
            # Применяем temperature sampling для более разнообразной генерации
            output = output.div(temperature)
            probs = torch.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probs, 1)[0]
            
            next_char = dataset.idx_to_char[next_char_idx.item()]
            generated_text += next_char
            
            input_seq = torch.tensor([[next_char_idx.item()]], dtype=torch.long)

    return generated_text

def train_model(model, dataset, num_epochs, batch_size, seq_length, lr):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        hidden = model.init_hidden(batch_size)
        
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            hidden = tuple([h.detach() for h in hidden])
            
            if x.size(0) != batch_size:
                hidden = model.init_hidden(x.size(0))
                
            output, hidden = model(x, hidden)
            loss = criterion(output, y)
            
            loss.backward()
            # Градиентный клиппинг для предотвращения взрыва градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f'Эпоха {epoch + 1}/{num_epochs}, Средняя потеря: {avg_loss:.4f}')
        
        # Сохраняем лучшую модель
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    # Параметры модели
    hidden_size = 512
    num_layers = 3
    dropout = 0.5
    seq_length = 100
    batch_size = 128
    num_epochs = 50
    lr = 0.001
    temperature = 0.8

    # Загрузка и подготовка данных
    texts = []
    for filename in os.listdir('data'):
        if filename.endswith('.txt'):
            with open(os.path.join('data', filename), "r", encoding="utf-8") as file:
                texts.append(file.read())

    dataset = TextDataset(texts, seq_length)
    model = CharRNN(len(dataset.chars), hidden_size, num_layers, dropout)
    
    # Обучение модели
    train_model(model, dataset, num_epochs, batch_size, seq_length, lr)
    
    # Генерация текста
    while True:
        try:
            user_input = input("Введите начальную строку (или 'exit' для выхода): ")
            if user_input.lower() == 'exit':
                break
                
            length = int(input("Введите длину генерируемого текста: "))
            generated_text = generate_text(model, dataset, user_input, length, temperature)
            if generated_text:
                print("\nСгенерированный текст:")
                print(generated_text)
            
        except ValueError:
            print("Ошибка: Введите корректное число для длины текста")