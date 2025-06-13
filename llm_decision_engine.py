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

def select_strategy_and_params(market_regime: str, lstm_signal: int, strategy_metrics: dict, param_grids: dict, is_forced_trade_scenario: bool = False) -> dict:
    """
    請 Gemini 直接建議「策略名稱+參數組合」，或在強制交易情境下建議行動。
    回傳格式:
    - 常規: {"strategy": "TrendStrategy", "params": {"rsi_low": 40, "rsi_high": 70}}
    - 強制交易: {"strategy": "ForcedBuy"} 或 {"strategy": "ForcedShort"} 或 {"strategy": "AbstainForceTrade"}
    """
    strategy_text = format_strategy_metrics(strategy_metrics)
    signal_text = {1: "預測上漲", -1: "預測下跌", 0: "無明確趨勢"}.get(lstm_signal, "無明確趨勢")
    valid_strategies = list(strategy_metrics.keys())

    if is_forced_trade_scenario:
        # 強制交易情境下的 Prompt
        example_forced_json = """{\n  "strategy": "ForcedBuy"\n}"""
        prompt = f"""你是一位專業的量化交易顧問。目前處於「強制交易決策」情境，因為已經連續多日沒有交易。

請根據以下市場資訊，決定是否進行強制交易，以及交易方向（買入或做空），或者選擇不進行強制交易。

**你的回覆必須是一個「單獨的 JSON 物件」**，不包含任何其他文字、註解、Markdown 標記 (例如 ```json) 或換行符號在 JSON 物件之外。

**嚴格的輸出格式要求：**
```json
{{\n  "strategy": "行動名稱"\n}}
```

**可選的 "strategy" 行動名稱 (你的 "strategy" 必須從此清單中選擇)：**
- "ForcedBuy" (強制買入)
- "ForcedShort" (強制做空)
- "AbstainForceTrade" (本次不進行強制交易)

**市場型態：** {market_regime}
**LSTM 預測：** {signal_text}

請根據上述所有資訊，選擇一個最適合的行動。
**再次強調，你的回覆「只能」是一個 JSON 物件，像這樣 (範例為強制買入)：**
{example_forced_json}
"""
    else:
        # 原有的常規策略選擇 Prompt
        param_text = "\n".join(
            f"- {name}: 可調整參數有 {list(param_grids.get(name, {}).keys())}" for name in valid_strategies
        )
        example_json = """{\n  "strategy": "TrendStrategy",\n  "params": {\n    "rsi_low": 40,\n    "rsi_high": 70\n  }\n}"""
        prompt = f"""你是一位專業的量化交易顧問。請根據以下市場資訊，直接建議一組「策略名稱」與「參數組合」。

你的回覆必須是一個「單獨的 JSON 物件」，不包含任何其他文字、註解、Markdown 標記 (例如 ```json) 或換行符號在 JSON 物件之外。

**嚴格的輸出格式要求：**
```json
{{\n  "strategy": "策略名稱",\n  "params": {{\n    "參數名稱1": 數值1,\n    "參數名稱2": 數值2\n  }}\n}}
```

**可選策略清單 (你的 "strategy" 必須從此清單中選擇)：**
{', '.join(valid_strategies)}

**各策略可調整的參數名稱提示：**
{param_text}

**市場型態：** {market_regime}
**LSTM 預測：** {signal_text}

**可選策略與其歷史績效參考：**
{strategy_text}

請根據上述所有資訊，選擇一個最適合的策略及其參數。
**再次強調，你的回覆「只能」是一個 JSON 物件，像這樣：**
{example_json}
"""

    if model is None:
        print("[警告] 未提供 GEMINI_API_KEY，已使用預設策略/行動")
        if is_forced_trade_scenario:
            return {"strategy": "AbstainForceTrade"} # 強制交易預設為不動作
        else:
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
            # 驗證回傳結果
            if isinstance(result, dict) and "strategy" in result:
                if is_forced_trade_scenario:
                    if result["strategy"] in ["ForcedBuy", "ForcedShort", "AbstainForceTrade"]:
                        return result # 對於強制交易，params 不是必需的
                else:
                    if result["strategy"] in valid_strategies and isinstance(result.get("params", {}), dict):
                        return result
        except json.JSONDecodeError as e_json:
            print(f"[警告] Gemini 回傳 JSON 解析失敗 ({e_json})：'{cleaned_response_text}'，已使用預設策略/行動")
        
        # 如果解析失敗或驗證失敗
        print(f"[警告] Gemini 回傳無效格式：'{response_text}'，已使用預設策略/行動")
        if is_forced_trade_scenario:
            return {"strategy": "AbstainForceTrade"}
        else:
            default_strategy = valid_strategies[0]
            return {"strategy": default_strategy, "params": {k: v[0] for k, v in param_grids.get(default_strategy, {}).items()}}

    except Exception as e:
        print(f"[錯誤] Gemini API 呼叫失敗：{e}")
        if is_forced_trade_scenario:
            return {"strategy": "AbstainForceTrade"}
        else:
            # 緊急 fallback (與原碼類似)
            if not valid_strategies and param_grids:
                 valid_strategies = list(param_grids.keys())
                 if not valid_strategies:
                      print("[嚴重錯誤] 無法確定預設策略，param_grids 為空。")
                      return {"strategy": "Error", "params": {}}
            default_strategy = valid_strategies[0]
            return {"strategy": default_strategy, "params": {k: v[0] for k, v in param_grids.get(default_strategy, {}).items()}}
