# 策略專案簡介（Strategy Project）

本儲存庫為一個簡易的量化交易策略回測框架，結合經典技術指標策略與基於 Gemini 大型語言模型（LLM）的決策引擎。可用於模擬不同市場狀況下的策略表現，並產出完整的績效報告與交易紀錄。

---

## 📦 環境建置說明（Environment Setup）

### 1. Python 版本要求
請安裝 Python 3.10 或以上版本。

### 2. 安裝必要套件
使用下列指令安裝專案所需套件：

```bash
pip install -r requirements.txt
```

依賴套件包含：
- `pandas`
- `numpy`
- `requests`
- `openpyxl`
- `yfinance`
- `scikit-learn`
- `torch`
- `google-generativeai`
- `python-dotenv`

---

## 🔑 設定 Gemini API 金鑰

請於專案根目錄建立 `.env` 檔案，並加入以下內容（請將 `your_key` 替換為實際金鑰）：

```env
GEMINI_API_KEY=your_key
```

---

## 🚀 執行範例（Running the Example）

完成環境建置與金鑰設定後，可使用以下指令啟動主程式：

```bash
python main.py
```

執行後，系統將自動完成以下任務：

1. 從 Yahoo Finance 擷取指定標的的歷史資料
2. 使用 LSTM 模型訓練並預測未來市場趨勢
3. 根據市場狀態選擇適用的策略模組進行交易模擬
4. 輸出每月績效報告（Excel 格式）與完整交易紀錄（CSV 格式）

---

## ⚙️ 進階設定

如需自訂以下內容，請參閱並修改 `config.py`：

- **回測起訖日期、股票代碼**：設定回測的時間範圍與標的。
- **INITIAL_CAPITAL**：初始資金（TWD）。
- **STOP_LOSS_THRESHOLD**：停損線（如 0.05 代表 5%）。
- **ALLOW_SHORT_SELLING**：是否允許放空。
- **SHORT_QTY_CAP**：放空時的最大數量。
- **LSTM_TRAIN_WINDOW**：LSTM 訓練所需資料點數。
- **LSTM_LOOKBACK_DAYS**：LSTM 觀察天數。
- **LSTM_PREDICT_DAYS**：LSTM 預測未來天數。
- **LSTM_RETRAIN_INTERVAL**：每幾天重訓一次 LSTM。
- **LSTM_EPOCHS**：LSTM 訓練 epoch 數。
- **LSTM_LEARNING_RATE**：LSTM 學習率。
- **策略選擇與評估依據**：如夏普值、勝率、報酬率。
- **ENABLE_STRATEGY_OPTIMIZATION**：是否啟用策略參數優化。
- **強制交易相關參數**：如最大無交易天數、強制交易資金分配、停利停損設定。
- **其他策略與回測細節參數**：可依需求擴充。

> 各參數皆有註解說明，請依需求調整，靈活打造屬於你的量化交易系統。

---

本框架設計模組化，方便日後擴充更多策略模組與市場預測方法，適用於個人研究、教育用途或策略開發實驗。