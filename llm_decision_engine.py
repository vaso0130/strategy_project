import os
import google.generativeai as genai
from google.api_core import retry
from dotenv import load_dotenv
import json

# 載入環境變數
load_dotenv()

# 初始化 Gemini Flash 模型
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("[資訊] 已成功初始化 Gemini 1.5 Flash 模型")
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

def select_strategy_and_params(market_regime: str, lstm_signal: int, strategy_metrics: dict, param_grids: dict) -> dict:
    """
    請 Gemini 直接建議「策略名稱+參數組合」。
    回傳格式: {"strategy": "TrendStrategy", "params": {"rsi_low": 40, "rsi_high": 70}}
    """
    strategy_text = format_strategy_metrics(strategy_metrics)
    signal_text = {1: "預測上漲", -1: "預測下跌", 0: "無明確趨勢"}.get(lstm_signal, "無明確趨勢")
    valid_strategies = list(strategy_metrics.keys())
    param_text = "\n".join(
        f"{name} 可調參數: {list(param_grids.get(name, {}).keys())}" for name in valid_strategies
    )

    prompt = f"""
你是一位專業的量化交易顧問，請根據以下市場資訊，直接建議一組「策略名稱」與「參數組合」。
請嚴格按照以下格式回覆（不要多加說明）：

{{
  "strategy": "策略名稱（只能從下列清單選一個）",
  "params": {{參數名稱: 數值, ...}}
}}

市場型態：{market_regime}
LSTM 預測：{signal_text}

可選策略與績效如下：
{strategy_text}

各策略可調參數如下：
{param_text}

請只回覆一組 JSON 格式（不要多加說明），例如：
{{"strategy": "TrendStrategy", "params": {{"rsi_low": 40, "rsi_high": 70}}}}
"""

    if model is None:
        print("[警告] 未提供 GEMINI_API_KEY，已使用預設策略")
        default_strategy = valid_strategies[0]
        return {"strategy": default_strategy, "params": {k: v[0] for k, v in param_grids.get(default_strategy, {}).items()}}

    try:
        no_retry = retry.Retry(predicate=lambda exc: False)
        response = model.generate_content(
            prompt,
            request_options={"timeout": 10, "retry": no_retry},
        )
        response_text = response.text.strip()
        try:
            result = json.loads(response_text)
            if (
                isinstance(result, dict)
                and "strategy" in result
                and result["strategy"] in valid_strategies
                and isinstance(result.get("params", {}), dict)
            ):
                return result
        except Exception:
            pass
        print(f"[警告] Gemini 回傳無效格式：{response_text}，已使用預設策略")
        default_strategy = valid_strategies[0]
        return {"strategy": default_strategy, "params": {k: v[0] for k, v in param_grids.get(default_strategy, {}).items()}}
    except Exception as e:
        print(f"[錯誤] Gemini API 呼叫失敗：{e}")
        default_strategy = valid_strategies[0]
        return {"strategy": default_strategy, "params": {k: v[0] for k, v in param_grids.get(default_strategy, {}).items()}}
