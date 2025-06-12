import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # 取最後時間步的輸出
        out = self.fc(out)
        return self.activation(out)

class LSTMPredictor:
    def __init__(self, lookback_days=30, predict_days=10, lr=0.001, epochs=20):
        self.lookback = lookback_days
        self.predict_days = predict_days
        self.scaler = MinMaxScaler()
        self.model = LSTMModel()
        self.epochs = epochs
        self.lr = lr

        # 嘗試使用 GPU，若失敗則改用 CPU
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.model.to(self.device)
            else:
                raise RuntimeError("CUDA not available")
        except:
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def preprocess(self, prices):
        prices = prices.copy()
        # scaled = self.scaler.fit_transform(prices[['close']])
        scaled = self.scaler.fit_transform(prices[['Close']]) # 改為大寫 'Close'
        X, y = [], []
        for i in range(len(scaled) - self.lookback - self.predict_days):
            seq_x = scaled[i:i+self.lookback]
            future_avg = scaled[i+self.lookback:i+self.lookback+self.predict_days].mean()
            current_price = scaled[i+self.lookback-1]
            label = 1 if future_avg > current_price else -1
            X.append(seq_x)
            y.append([label])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)

    def train(self, prices):
        X, y = self.preprocess(prices)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            break # 簡化訓練流程，僅跑一個 epoch 以縮短時間

    def predict(self, recent_prices):
        if len(recent_prices) < self.lookback:
            return 0

        self.model.eval()
        prices = recent_prices.copy()
        # data = prices['close'].values[-self.lookback:]
        data = prices['Close'].values[-self.lookback:] # 改為大寫 'Close'
        data_scaled = self.scaler.transform(
            # pd.DataFrame(data.reshape(-1, 1), columns=['close'])
            pd.DataFrame(data.reshape(-1, 1), columns=['Close']) # 改為大寫 'Close'
        )
        input_tensor = torch.tensor(data_scaled.reshape(1, self.lookback, 1), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.model(input_tensor).item()

        # 降低判定門檻讓 up/down 訊號更容易出現
        if pred > 0.05:
            return 1
        elif pred < -0.05:
            return -1
        else:
            return 0
