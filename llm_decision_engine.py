import os
import google.generativeai as genai
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 初始化 Gemini Flash 模型
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

def format_strategy_metrics(strategy_metrics: dict) -> str:
    lines = []
    for name, metrics in strategy_metrics.items():
        bias = metrics.get('bias', 0)
        bias_text = '多' if bias == 1 else '空' if bias == -1 else '中性'
        line = f"{name}: Sharpe={metrics.get('sharpe', 0):.2f}, 勝率={metrics.get('win_rate', 0):.2f}, 偏向={bias_text}"
        lines.append(line)
    return "\n".join(lines)

def select_strategy(market_regime: str, lstm_signal: int, strategy_metrics: dict) -> str:
    """
    使用 Gemini 1.5 Flash 判斷最適策略。失敗時會 fallback 回預設策略。
    """
    strategy_text = format_strategy_metrics(strategy_metrics)
    signal_text = {1: "預測上漲", -1: "預測下跌", 0: "無明確趨勢"}.get(lstm_signal, "無明確趨勢")
    valid_strategies = list(strategy_metrics.keys())
    strategy_list = "\n".join(f"- {s}" for s in valid_strategies)

    prompt = f"""
你是一位專業的量化交易顧問，請根據以下市場資訊，**只從下列清單中挑選一個最適合的策略名稱**。請注意：

- 請只回覆清單中的一個「策略名稱」
- 請不要多加說明或額外語句
- 請不要回答「無法判斷」、「建議觀望」等語句

市場型態：{market_regime}
LSTM 預測：{signal_text}

可選策略與績效如下：
{strategy_text}

請從以下策略中選出一個最適合的（只能選一個）：
{strategy_list}

請只回答策略名稱，例如：TrendStrategy
"""

    if model is None:
        print("[警告] 未提供 GEMINI_API_KEY，已使用預設策略")
        return valid_strategies[0]

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        if response_text in valid_strategies:
            return response_text
        else:
            print(f"[警告] Gemini 回傳無效策略名稱：{response_text}，已使用預設策略")
            return valid_strategies[0]  # fallback 預設策略

    except Exception as e:
        print(f"[錯誤] Gemini API 呼叫失敗：{e}")
        return valid_strategies[0]  # fallback 預設策略
