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
        
        # 強化版清理：嘗試提取第一個 '{' 和最後一個 '}' 之間的內容
        try:
            # 先移除常見的 Markdown 標記 (```json ... ``` or ``` ... ```)
            if response_text.startswith("```") and response_text.endswith("```"):
                response_text = response_text[3:-3].strip()
                # 如果是 ```json ... ```，移除 'json'
                if response_text.lower().startswith("json"):
                    response_text = response_text[4:].strip()
            
            # 再移除單獨的 "json" 前綴 (以防萬一)
            if response_text.lower().startswith("json"):
                response_text = response_text[4:].strip()

            # 提取 '{' 和 '}' 之間的內容
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start != -1 and json_end != 0 and json_start < json_end:
                cleaned_response_text = response_text[json_start:json_end]
            else:
                # 如果找不到有效的 JSON 結構，則使用原始清理後的文本（可能導致解析失敗）
                cleaned_response_text = response_text 
        except Exception: # 如果提取過程中出錯，還是用原始 strip 後的文本
            cleaned_response_text = response_text.strip("` \n") # Fallback to simpler stripping

        try:
            result = json.loads(cleaned_response_text)
            if (
                isinstance(result, dict)
                and "strategy" in result
                and result["strategy"] in valid_strategies
                and isinstance(result.get("params", {}), dict)
            ):
                return result
        except json.JSONDecodeError as e_json:
            print(f"[警告] Gemini 回傳 JSON 解析失敗 ({e_json})：'{cleaned_response_text}'，已使用預設策略")
            # Fall through to default strategy
        
        # 如果解析失敗或驗證失敗，則執行到這裡
        # print(f"[警告] Gemini 回傳無效格式（驗證失敗或解析後非預期）：'{response_text}'，已使用預設策略") # 可以調整此處的 log
        default_strategy = valid_strategies[0]
        return {"strategy": default_strategy, "params": {k: v[0] for k, v in param_grids.get(default_strategy, {}).items()}}
    except Exception as e:
        print(f"[錯誤] Gemini API 呼叫失敗：{e}")
        default_strategy = valid_strategies[0] # valid_strategies 可能在此處未定義，如果 API 呼叫在 prompt 建立前失敗
        if not valid_strategies and param_grids: # 緊急 fallback
             valid_strategies = list(param_grids.keys())
             if not valid_strategies: # 如果連 param_grids 都沒有，就無法提供預設策略了
                  print("[嚴重錯誤] 無法確定預設策略，param_grids 為空。")
                  return {"strategy": "Error", "params": {}} # 或者拋出異常
             default_strategy = valid_strategies[0]

        return {"strategy": default_strategy, "params": {k: v[0] for k, v in param_grids.get(default_strategy, {}).items()}}
