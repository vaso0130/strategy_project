import time
import requests
import pandas as pd

def download_yahoo_data(symbol: str, start_date: str, end_date: str) -> dict:
    """
    從 Yahoo Finance JSON API 下載股價資料。
    """
    period1 = int(time.mktime(time.strptime(start_date, "%Y-%m-%d")))
    period2 = int(time.mktime(time.strptime(end_date, "%Y-%m-%d")))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={period1}&period2={period2}&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"下載失敗：HTTP {response.status_code}，內容：{response.text[:200]}")

    return response.json()

def parse_yahoo_chart_to_df(json_data: dict) -> pd.DataFrame:
    """
    將 Yahoo JSON 結果轉換為完整價格 DataFrame。
    包含：date, open, high, low, close, adjclose, volume
    """
    result = json_data['chart']['result'][0]
    timestamps = result['timestamp']
    quote = result['indicators']['quote'][0]
    adjclose = result['indicators']['adjclose'][0]['adjclose']

    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit='s'),
        "open": quote.get('open'),
        "high": quote.get('high'),
        "low": quote.get('low'),
        "close": quote.get('close'),
        "adjclose": adjclose,
        "volume": quote.get('volume')
    })

    df = df.dropna(subset=['adjclose'])  # 移除沒有價格的行
    df = df.sort_values("date").reset_index(drop=True)

    # 若有欄位遺失，報錯
    required_cols = ['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("資料欄位不完整")

    return df

def load_price_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    整合下載與轉換流程，回傳乾淨的歷史價格資料（DataFrame）。
    """
    try:
        raw_json = download_yahoo_data(symbol, start_date, end_date)
        return parse_yahoo_chart_to_df(raw_json)
    except Exception as e:
        print(f"[錯誤] 無法載入資料：{e}")
        raise
