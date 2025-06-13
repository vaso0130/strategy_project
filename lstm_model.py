import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

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
        self.is_trained = False  # Initialize is_trained status
        # Force CPU to avoid CUDA errors if environment is not perfectly configured
        self.device = torch.device("cpu") 
        print(f"[資訊] LSTMPredictor 強制使用裝置: {self.device}")
        self.model.to(self.device)
        # print(f"[資訊] LSTMPredictor 使用裝置: {self.device}")

    def preprocess(self, prices_df: pd.DataFrame): # Expects a DataFrame with a 'Close' column
        if 'Close' not in prices_df.columns:
            raise ValueError("DataFrame must contain a 'Close' column for preprocessing.")
        
        prices = prices_df['Close'].values.reshape(-1, 1)
        # Fit scaler only on training data, transform on new data.
        # For simplicity here, we fit_transform, assuming preprocess is called by train.
        # If preprocess were also called by predict, scaler fitting needs careful handling.
        scaled = self.scaler.fit_transform(prices) 
        X, y = [], []
        # Ensure loop range is valid
        # Loop up to the point where there's enough data for a lookback window AND a subsequent predict_days target
        for i in range(len(scaled) - self.lookback - self.predict_days + 1):
            X.append(scaled[i:(i + self.lookback)])
            # Target is the average of the prices over the predict_days horizon
            y.append(scaled[i + self.lookback : i + self.lookback + self.predict_days].mean())
        
        if not X or not y: # Handle case where X or y is empty
            return None, None
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    def train(self, prices_df: pd.DataFrame): # Expects a DataFrame with 'date' and 'Close'
        if 'Close' not in prices_df.columns:
            print("[錯誤] LSTMPredictor.train: 'Close' 欄位不存在於傳入的 DataFrame。")
            return # self.is_trained remains as it was

        # Minimum data points needed to form at least one (X,y) pair
        min_data_len = self.lookback + self.predict_days
        if len(prices_df) < min_data_len:
            print(f"[警告] LSTMPredictor.train: 訓練資料不足 ({len(prices_df)} 筆)，至少需要 {min_data_len} 筆才能產生一個樣本。跳過此次訓練。")
            return # self.is_trained remains as it was

        X, y = self.preprocess(prices_df)
        
        if X is None or y is None or X.nelement() == 0 or y.nelement() == 0:
            print(f"[警告] LSTMPredictor.train: 預處理後無有效訓練數據。原始數據 {len(prices_df)} 筆。跳過此次訓練。")
            return # self.is_trained remains as it was

        dataset = TensorDataset(X, y)
        batch_size = min(32, len(X)) # Ensure batch_size is not greater than the number of samples
        if batch_size == 0: # Should be caught by X.nelement() == 0, but as a safeguard
            print(f"[警告] LSTMPredictor.train: 預處理後無有效樣本可供訓練 (X 為空)。跳過此次訓練。")
            return

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train() # Set model to training mode
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            # print(f"Epoch {epoch+1}/{self.epochs}, Avg Loss: {epoch_loss/num_batches:.4f}")
        
        self.is_trained = True # Set to True only after successful training completion
        print(f"[資訊] LSTMPredictor.train: 模型訓練完成。is_trained = {self.is_trained}")

    def predict(self, recent_prices_df: pd.DataFrame): # Expects a DataFrame with 'Close'
        if not self.is_trained:
            print("[警告] LSTMPredictor.predict: 模型尚未訓練，無法預測。返回中性信號 0。")
            return 0 

        if 'Close' not in recent_prices_df.columns:
            print("[錯誤] LSTMPredictor.predict: 'Close' 欄位不存在於傳入的 DataFrame。返回中性信號 0。")
            return 0
            
        if len(recent_prices_df) < self.lookback:
            print(f"[警告] LSTMPredictor.predict: 預測所需數據不足 ({len(recent_prices_df)} 筆)，至少需要 {self.lookback} 筆。返回中性信號 0。")
            return 0

        self.model.eval() # Set model to evaluation mode
        
        # Extract 'Close' prices, ensure it's a 2D array, and scale using the *already fitted* scaler
        prices_values = recent_prices_df['Close'].values[-self.lookback:].reshape(-1, 1)
        try:
            # Scaler expects to be fitted. If predict is called before train, this will fail.
            # The self.is_trained check should prevent this, but good to be aware.
            scaled_prices = self.scaler.transform(prices_values) 
        except Exception as e:
            print(f"[錯誤] LSTMPredictor.predict: 數據縮放失敗 ({e})。可能 scaler 未被 fit。返回中性信號 0。")
            return 0

        X_pred = torch.tensor(scaled_prices.reshape(1, self.lookback, 1), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            prediction_scaled = self.model(X_pred)
        
        try:
            prediction_inverted = self.scaler.inverse_transform(prediction_scaled.cpu().numpy())
        except Exception as e:
            print(f"[錯誤] LSTMPredictor.predict: 預測結果反向縮放失敗 ({e})。返回中性信號 0。")
            return 0
        
        # Convert prediction to signal: 1 for up, -1 for down, 0 for neutral
        # Compare with the last known close price from the input data
        last_close_price = recent_prices_df['Close'].iloc[-1]
        predicted_value = prediction_inverted[0][0]

        # Define a threshold for significant change, e.g., 0.2%
        threshold_percentage = 0.002 # Changed from 0.005 to 0.002
        price_diff = predicted_value - last_close_price
        percentage_diff = price_diff / last_close_price

        if percentage_diff > threshold_percentage:
            return 1  # Predict up
        elif percentage_diff < -threshold_percentage:
            return -1 # Predict down
        else:
            return 0  # Predict neutral
