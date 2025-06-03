import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from joblib import dump, load

# ==================================================
# 1. Улучшенная обработка данных и подготовка
# ==================================================

# Загрузка данных
df_prices = pd.read_csv('Adj Close.csv', parse_dates=['date'])

# Предварительная обработка
companies = ["Tesla", "Apple", "Amazon", "Nvidia", "Microsoft"]
sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
scalers = {}  # Для сохранения скейлеров

def prepare_data():
    dfs = []
    for company in companies:
        # Загрузка данных
        prices = df_prices[['date', company]].copy()
        sentiments = pd.read_csv(f'{company}_sentiment.csv', parse_dates=['date'])
        
        # Преобразование и агрегация
        sentiments['sentiment_score'] = sentiments['sentiment'].map(sentiment_map)
        daily_sent = sentiments.groupby('date')['sentiment_score'].mean().reset_index()
        
        # Слияние данных
        merged = pd.merge(prices, daily_sent, on='date', how='left')
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
        merged['company'] = company
        merged.rename(columns={company: 'price'}, inplace=True)
        dfs.append(merged)
    
    df = pd.concat(dfs).sort_values('date').reset_index(drop=True)
    
    # Стандартизация по компаниям
    df['price_scaled'] = 0.0
    df['sentiment_scaled'] = 0.0
    
    for company in companies:
        company_mask = df['company'] == company
        price_scaler = StandardScaler()
        sent_scaler = StandardScaler()
        
        df.loc[company_mask, 'price_scaled'] = price_scaler.fit_transform(
            df.loc[company_mask, ['price']])
        df.loc[company_mask, 'sentiment_scaled'] = sent_scaler.fit_transform(
            df.loc[company_mask, ['sentiment_score']])
        
        # Сохраняем скейлеры для обратного преобразования
        scalers[f'{company}_price'] = price_scaler
        scalers[f'{company}_sent'] = sent_scaler
    
    # Кодирование компаний
    company_encoder = {company: idx for idx, company in enumerate(companies)}
    df['company_id'] = df['company'].map(company_encoder)
    
    return df

df = prepare_data()
dump(scalers, 'scalers.joblib')  # Сохраняем скейлеры

# ==================================================
# 2. Улучшенное создание последовательностей
# ==================================================

def print_lr(optimizer, epoch):
    """Печатает текущий learning rate"""
    lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}: Current learning rate = {lr:.2e}')

def create_sequences(data, window_size=10):
    sequences, targets, company_ids = [], [], []
    
    for company_id in data['company_id'].unique():
        company_data = data[data['company_id'] == company_id]
        company_data = company_data.sort_values('date').reset_index(drop=True)
        
        # Создаем массивы значений
        prices = company_data['price_scaled'].values
        sentiments = company_data['sentiment_scaled'].values
        cids = company_data['company_id'].values
        
        for i in range(len(company_data) - window_size):
            # Комбинируем признаки
            seq = np.column_stack((
                prices[i:i+window_size],
                sentiments[i:i+window_size],
                cids[i:i+window_size]
            ))
            
            sequences.append(seq)
            targets.append(prices[i+window_size])  # Цель - следующий день
            company_ids.append(company_id)
    
    return np.array(sequences), np.array(targets), np.array(company_ids)

window_size = 20  # Увеличили размер окна
X, y, company_ids = create_sequences(df, window_size)

# ==================================================
# 3. Корректное разделение данных (временные ряды)
# ==================================================

# Разделяем по времени (не перемешивая)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Конвертация в тензоры PyTorch
train_data = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
)
val_data = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
)
test_data = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
)

# DataLoader с учетом временных рядов
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ==================================================
# 4. Улучшенная архитектура модели
# ==================================================

class EnhancedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        output, _ = self.gru(x)
        
        # Механизм внимания
        attention_weights = self.attention(output)
        context_vector = torch.sum(attention_weights * output, dim=1)
        
        # Нормализация
        context_vector = self.layer_norm(context_vector)
        
        return self.fc(context_vector)

# Параметры модели
input_size = 3  # price_scaled + sentiment_scaled + company_id
model = EnhancedGRU(
    input_size=input_size,
    hidden_size=128,  # Увеличили скрытый слой
    num_layers=3,
    dropout=0.4
)

# ==================================================
# 5. Улучшенный процесс обучения с валидацией
# ==================================================

# Оптимизатор и планировщик
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
criterion = nn.HuberLoss()

# ИСПРАВЛЕННЫЙ планировщик (без verbose)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    threshold=0.0001,
    threshold_mode='rel',
    cooldown=0,
    min_lr=1e-6
)

def train_model(model, epochs=200):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    patience_counter = 0
    
    for epoch in range(epochs):
        # Тренировка
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
        
        # Нормализация потерь
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Обновление LR с логированием
        scheduler.step(val_loss)
        if epoch % 10 == 0:
            print_lr(optimizer, epoch)
        
        # Ранняя остановка
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model, history

trained_model, training_history = train_model(model)

# ==================================================
# 6. Улучшенная оценка и визуализация
# ==================================================

def inverse_transform(company, values):
    """Обратное преобразование нормализованных значений"""
    scaler = scalers[f'{company}_price']
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()

def evaluate_model(model):
    model.eval()
    all_preds = []
    all_targets = []
    company_results = {company: {'preds': [], 'targets': []} for company in companies}
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            outputs = model(X_test)
            all_preds.append(outputs.numpy())
            all_targets.append(y_test.numpy())
    
    all_preds = np.vstack(all_preds).flatten()
    all_targets = np.vstack(all_targets).flatten()
    
    # Рассчет RMSE в оригинальном масштабе
    test_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    print(f"Test RMSE (scaled): {test_rmse:.4f}")
    r2 = r2_score(all_targets, all_preds)
    print(f"Test R2 (scaled): {r2:.4f}")


    # Визуализация по компаниям
    plt.figure(figsize=(15, 10))
    for idx, company in enumerate(companies):
        company_mask = (company_ids[-len(all_targets):] == idx)
        if np.any(company_mask):
            company_preds = all_preds[company_mask]
            company_targets = all_targets[company_mask]
            
            # Обратное преобразование
            orig_preds = inverse_transform(company, company_preds)
            orig_targets = inverse_transform(company, company_targets)
            
            # Рассчет RMSE для компании
            company_rmse = np.sqrt(mean_squared_error(orig_targets, orig_preds))
            
            # График
            plt.subplot(3, 2, idx+1)
            plt.plot(orig_targets, label='Actual', alpha=0.7)
            plt.plot(orig_preds, label='Predicted', alpha=0.7)
            plt.title(f"{company} | RMSE: {company_rmse:.2f}")
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('predictions_comparison.png')
    plt.show()
    
    # Loss history
    plt.figure(figsize=(10, 5))
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()

evaluate_model(trained_model)