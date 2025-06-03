import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class EnhancedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
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
        attention_weights = self.attention(output)
        context_vector = torch.sum(attention_weights * output, dim=1)
        context_vector = self.layer_norm(context_vector)
        return self.fc(context_vector)

def load_intel_data():
    """Загрузка и подготовка данных Intel"""
    try:
        intel_prices = pd.read_csv('Adj Close.csv', parse_dates=['date'])
        intel_prices = intel_prices[['date', 'Intel']].rename(columns={'Intel': 'price'})
    except FileNotFoundError:
        raise FileNotFoundError("Файл с ценами 'Adj Close.csv' не найден")
    
    # Обработка сентимента
    try:
        intel_sent = pd.read_csv('Intel_sentiment.csv', parse_dates=['date'])
        intel_sent['sentiment_score'] = intel_sent['sentiment'].map(
            {'negative': -1, 'neutral': 0, 'positive': 1})
        daily_sent = intel_sent.groupby('date')['sentiment_score'].mean().reset_index()
        merged = pd.merge(intel_prices, daily_sent, on='date', how='left')
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
    except FileNotFoundError:
        merged = intel_prices.copy()
        merged['sentiment_score'] = 0  # Заполняем нулями если нет данных
    
    # Стандартизация
    price_scaler = StandardScaler()
    sent_scaler = StandardScaler()
    
    merged['price_scaled'] = price_scaler.fit_transform(merged[['price']])
    merged['sentiment_scaled'] = sent_scaler.fit_transform(merged[['sentiment_score']])
    
    return merged, price_scaler
# 3. Создание последовательностей для Intel
def create_intel_sequences(data, window_size=20):
    sequences, targets = [], []
    for i in range(len(data) - window_size):
        seq = data.iloc[i:i+window_size][['price_scaled', 'sentiment_scaled']].values
        # Добавляем фиктивный третий признак (нули)
        seq = np.pad(seq, ((0,0),(0,1)), mode='constant') 
        sequences.append(seq)
        targets.append(data.iloc[i+window_size]['price_scaled'])
    return np.array(sequences), np.array(targets)

# 4. Загрузка модели
def load_trained_model(input_size=3, hidden_size=128):  # Было 2, стало 3
    model = EnhancedGRU(input_size=input_size, 
                      hidden_size=hidden_size, 
                      num_layers=3, 
                      dropout=0.4)
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# 5. Основной процесс тестирования
def test_intel_stock(window_size=20):
    # Загрузка данных
    intel_data, price_scaler = load_intel_data()
    
    # Создание последовательностей
    X, y = create_intel_sequences(intel_data, window_size)
    
    # Разделение на тестовые данные (последние 20%)
    test_size = int(0.2 * len(X))
    X_test, y_test = X[-test_size:], y[-test_size:]
    
    # Конвертация в тензоры
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Загрузка модели
    model = load_trained_model(input_size=3)  # Убедитесь, что передаете 3
    
    # Тестирование
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(y_batch.squeeze().tolist())
    
    # Преобразование в оригинальный масштаб
    pred_prices = price_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actual_prices = price_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # Расчет метрик
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mae = mean_absolute_error(actual_prices, pred_prices)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Intel Price', alpha=0.7)
    plt.plot(pred_prices, label='Predicted', alpha=0.7)
    plt.title(f'Intel Stock Price Prediction\nTest RMSE: {rmse:.2f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid()
    plt.savefig('intel_predictions.png')
    plt.show()
    
    return rmse, mae

if __name__ == "__main__":
    test_rmse, test_mae = test_intel_stock()
    print(f"Intel Stock Test Results:\nRMSE = {test_rmse:.2f}")