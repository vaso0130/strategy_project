# Strategy Project

This repository contains a simple trading strategy backtesting framework that combines classic
tactics with a Gemini LLM based decision engine.

## Environment Setup

1. **Python**: Install Python 3.10 or later.
2. **Dependencies**: Install required packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   The packages include `pandas`, `numpy`, `requests`, `openpyxl`, `yfinance`,
   `scikit-learn`, `torch` and `google-generativeai` ,`python-dotenv`.

## GEMINI_API_KEY
新增.env
寫入
GEMINI_API_KEY=you_key

## Running the Example

After installing the dependencies and setting the API key, run:

```bash
python main.py
```

The script downloads historical data from Yahoo Finance, trains an LSTM model
and outputs a monthly performance report.
