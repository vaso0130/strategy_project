import os
import google.generativeai as genai
from dotenv import load_dotenv
from config import LLM_FREE_ANALYSIS_DAYS, LLM_STRUCTURED_DECISION_DAYS, LSTM_PREDICT_DAYS, ALLOW_SHORT_SELLING, RISK_PROFILE
import pandas as pd

# 載入環境變數
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file or environment variables")
genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config=generation_config,
    safety_settings=safety_settings,
)

def parse_structured_response(response_text):
    """
    解析 LLM 結構化回傳格式，回傳 (strategy_name, params, capital_allocation_factor, protection_action, protection_reason)
    """
    import json, ast, re
    # 防呆：若 LLM 回傳為空或格式明顯錯誤，直接回傳 Abstain 並 log
    if not response_text or "{" not in response_text:
        print("[LLM FORMAT ERROR] LLM 回傳內容為空或格式錯誤，已自動Abstain")
        return "Abstain", None, None, None, None
    strategy_name = response_text.strip()
    params = None
    capital_allocation_factor = None
    protection_action = None
    protection_reason = None
    if "{" in response_text and "}" in response_text:
        name_part = response_text.split("{", 1)[0].strip()
        json_start_index = response_text.find("{")
        open_braces = 0
        json_end_index = -1
        if json_start_index != -1:
            for i, char in enumerate(response_text[json_start_index:]):
                if char == '{': open_braces += 1
                elif char == '}': open_braces -= 1
                if open_braces == 0:
                    json_end_index = json_start_index + i
                    break
        if json_end_index != -1:
            actual_json_str = response_text[json_start_index : json_end_index + 1]
            parsed_dict = None
            candidates_to_try = [actual_json_str]
            if actual_json_str.startswith("{{") and actual_json_str.endswith("}}") and len(actual_json_str) > 3:
                candidates_to_try.append(actual_json_str[1:-1])
            for s_idx, s_to_parse in enumerate(candidates_to_try):
                try:
                    parsed_dict = json.loads(s_to_parse)
                    if isinstance(parsed_dict, dict):
                        break
                    else:
                        parsed_dict = None
                except json.JSONDecodeError:
                    parsed_dict = None
                if parsed_dict is None:
                    try:
                        evaluated_result = ast.literal_eval(s_to_parse)
                        if isinstance(evaluated_result, dict):
                            parsed_dict = evaluated_result
                            break
                    except (ValueError, SyntaxError, TypeError):
                        pass
            if parsed_dict is not None:
                params = parsed_dict
                strategy_name = name_part
    # 解析資金分配
    factor_match = re.search(r"capital_allocation_factor:\s*([0-9.]+)", response_text, re.IGNORECASE)
    if factor_match:
        try:
            factor_val = float(factor_match.group(1))
            if 0.0 <= factor_val <= 1.0:
                capital_allocation_factor = factor_val
        except ValueError:
            pass
    # 解析風控
    prot_action_match = re.search(r"protection_action:\s*([A-Z_]+)", response_text)
    prot_reason_match = re.search(r"protection_reason:\s*([^,\n]+)", response_text)
    if prot_action_match:
        protection_action = prot_action_match.group(1).strip()
    if prot_reason_match:
        protection_reason = prot_reason_match.group(1).strip()
    return strategy_name, params, capital_allocation_factor, protection_action, protection_reason


def select_strategy_and_params(current_date, price_df, news_sentiment_summary, available_strategies, lstm_signal,
                               days_since_last_trade=0, max_days_no_trade=0, is_forced_trade_scenario=False,
                               recommended_strategy_name=None, recommended_strategy_signal=None,
                               current_active_strategy_name=None, current_strategy_days_active=0, min_strategy_duration=0,
                               last_trade_pnl=0.0, cumulative_pnl=0.0, optimizer_best_params=None, regime=None,
                               total_trades=0, win_rate=None):
    """
    main.py 不再傳遞 price_data_str，改傳 price_df（含所有歷史資料）和 current_date（字串或 datetime.date）。
    本函式內部根據 config 取對應天數資料組裝 price_data_str。
    """
    llm_decision_context = {"prompt": "", "response": ""}
    capital_allocation_factor = None
    # 取得當日日期型別
    if isinstance(current_date, str):
        current_date_dt = pd.to_datetime(current_date).date()
    else:
        current_date_dt = current_date
    # 內部切片資料
    def calculate_atr(df, window=14):
        # 計算 ATR，需有 High、Low、Close 欄位
        if not all(col in df.columns for col in ['High', 'Low', 'Close']):
            return pd.Series([None]*len(df), index=df.index)
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=1).mean()
        return atr

    def get_price_data_str(df, end_date, num_days):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.date
        # 確保 volume 欄位轉為大寫 Volume
        if 'volume' in df.columns and 'Volume' not in df.columns:
            df['Volume'] = df['volume']
        # 新增 ATR 欄位
        if not 'ATR' in df.columns and all(col in df.columns for col in ['High', 'Low', 'Close']):
            df['ATR'] = calculate_atr(df)
        mask = df['date'] < end_date
        sub_df = df[mask].tail(num_days)
        cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR'] if col in sub_df.columns]
        return sub_df[cols].to_string() if not sub_df.empty else "No historical price data available for LLM."
    if is_forced_trade_scenario:
        # 強制交易也自動組裝 price_data_str，預設用 LLM_STRUCTURED_DECISION_DAYS
        price_data_str_forced = get_price_data_str(price_df, current_date_dt, LLM_STRUCTURED_DECISION_DAYS)
        lstm_signal_str = "Buy" if lstm_signal == 1 else "Short" if lstm_signal == -1 else "Neutral"
        recommended_strategy_signal_str = "N/A" # Default if None
        if recommended_strategy_signal is not None:
            if recommended_strategy_signal == 1:
                recommended_strategy_signal_str = "Buy"
            elif recommended_strategy_signal == -1:
                recommended_strategy_signal_str = "Short"
            elif recommended_strategy_signal == 0:
                recommended_strategy_signal_str = "Neutral"
        prompt = f"""You are a sophisticated trading LLM.
Current Date: {current_date}
Price Data (last {LLM_STRUCTURED_DECISION_DAYS} days):
{price_data_str_forced}
Market News/Sentiment (summarized): {news_sentiment_summary}
LSTM Signal: {lstm_signal_str}
Days since last trade: {days_since_last_trade} / {max_days_no_trade} (Max)

The currently active strategy is '{recommended_strategy_name if recommended_strategy_name else "None"}' which today suggests a '{recommended_strategy_signal_str}' action.
This active strategy has been running for {current_strategy_days_active} days (min duration: {min_strategy_duration} days).

We have not traded for {days_since_last_trade} days. The active strategy has also met its minimum duration of {min_strategy_duration} days.
The system is now **strongly considering a 'forced trade'** to ensure market participation and capture potential opportunities arising from this prolonged inactivity.
Based on ALL available information (LSTM, the active strategy's signal, price action), the system has a **clear bias towards making a trade (ForcedBuy or ForcedShort).**

Evaluate the following options:
1.  'ForcedBuy': Is there any basis to expect an upward move or a reversal of a downtrend (e.g., oversold conditions, support levels)?
2.  'ForcedShort': Is there any basis to expect a downward move or a continuation of a downtrend (e.g., overbought conditions, resistance levels)?
3.  'AbstainForceTrade': Choose this **only as a last resort** if both ForcedBuy and ForcedShort appear exceptionally risky, contradict overwhelming evidence, or if market conditions are extremely uncertain.

Your decision must be one of: ForcedBuy, ForcedShort, AbstainForceTrade.
Provide a brief rationale for your choice, especially if abstaining.
"""
        llm_decision_context["prompt"] = prompt
        try:
            response = model.generate_content(prompt)
            decision_text = response.text.strip() # Changed variable name
            llm_decision_context["response"] = decision_text
            # Extract the core decision
            if "ForcedBuy" in decision_text:
                decision = "ForcedBuy"
            elif "ForcedShort" in decision_text:
                decision = "ForcedShort"
            elif "AbstainForceTrade" in decision_text: # Check for Abstain last
                decision = "AbstainForceTrade"
            else:
                print(f"LLM returned unclear forced trade decision: {decision_text}. Defaulting to AbstainForceTrade.")
                decision = "AbstainForceTrade"
            return decision, None, None, llm_decision_context # MODIFIED RETURN
        except Exception as e:
            print(f"Error calling Gemini API for forced trade: {e}")
            return "AbstainForceTrade", None, None, llm_decision_context # MODIFIED RETURN
    else:
        try:
            # 1. 先進行自由分析（取 config 設定天數）
            price_data_str_free = get_price_data_str(price_df, current_date_dt, LLM_FREE_ANALYSIS_DAYS)
            free_analysis_text, free_analysis_prompt = llm_free_analysis(
                current_date=current_date, price_data_str=price_data_str_free, news_sentiment_summary=news_sentiment_summary,
                lstm_signal=lstm_signal, regime=regime
            )
            print(f"[LLM Free Analysis Prompt] {free_analysis_prompt}")
            print(f"[LLM Free Analysis Result] {free_analysis_text}")
            # 2. 再進行結構化決策（取 config 設定天數）
            price_data_str_struct = get_price_data_str(price_df, current_date_dt, LLM_STRUCTURED_DECISION_DAYS)
            structured_response, structured_prompt = llm_structured_decision(
                current_date=current_date, price_data_str=price_data_str_struct, news_sentiment_summary=news_sentiment_summary,
                lstm_signal=lstm_signal, regime=regime, last_trade_pnl=last_trade_pnl, cumulative_pnl=cumulative_pnl,
                available_strategies=available_strategies, optimizer_best_params=optimizer_best_params,
                free_analysis_text=free_analysis_text, total_trades=total_trades, win_rate=win_rate,
                risk_profile=RISK_PROFILE
            )
            print(f"[LLM Structured Prompt] {structured_prompt}")
            print(f"[LLM Structured Result] {structured_response}")
            llm_decision_context = {
                "free_analysis_prompt": free_analysis_prompt,
                "free_analysis_text": free_analysis_text,
                "structured_prompt": structured_prompt,
                "structured_response": structured_response
            }
            # 3. 解析結構化回傳
            strategy_name, params, capital_allocation_factor, protection_action, protection_reason = parse_structured_response(structured_response)
            # 若格式異常，log 警告
            if not strategy_name or (params is None and "{" in structured_response):
                print(f"[LLM FORMAT ERROR] Could not parse structured response: {structured_response}")
            return strategy_name, params, capital_allocation_factor, llm_decision_context, protection_action, protection_reason
        except Exception as e:
            print(f"[LLM ERROR] Error calling Gemini API or parsing response for strategy selection: {e}")
            print(f"[LLM ERROR] Last LLM structured response: {locals().get('structured_response', '')}")
            return "Abstain", None, None, llm_decision_context, None, None


def llm_free_analysis(current_date, price_data_str, news_sentiment_summary, lstm_signal, regime):
    """
    第一次 LLM：僅用一段話描述市場趨勢，不限格式。
    """
    lstm_predict_day = LSTM_PREDICT_DAYS
    short_sell = ALLOW_SHORT_SELLING
    prompt = f"""
你是一位專業量化交易 LLM，請根據下列30天市場數據，僅用一段話描述目前市場趨勢與特徵（如多頭、空頭、盤整、波動性、量能變化等），
請勿僅重複風險管理、數據不足等模板建議，必須根據現有數據做出具體判斷。
[市場 regime 狀態] {regime}
Current Date: {current_date}
Price Data (last 30 days):
{price_data_str}
Market News/Sentiment: {news_sentiment_summary}
LSTM預測未來{lstm_predict_day}天的Signal: {lstm_signal} 
目前是否可以做空: {short_sell}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), prompt
    except Exception as e:
        print(f"Error calling LLM for free analysis: {e}")
        return "", prompt


def llm_structured_decision(current_date, price_data_str, news_sentiment_summary, lstm_signal, regime, last_trade_pnl, cumulative_pnl, available_strategies, optimizer_best_params, free_analysis_text, total_trades=0, win_rate=None, risk_profile=None):
    """
    第二次 LLM：根據第一次分析與原始數據，產生結構化決策。
    新增：total_trades, win_rate 參數
    """
    win_rate_str = f"{win_rate:.2%}" if win_rate is not None else "N/A"
    lstm_predict_day = LSTM_PREDICT_DAYS
    short_sell = ALLOW_SHORT_SELLING
    risk_profile = risk_profile or RISK_PROFILE
    risk_profile_desc = {
        'aggressive': '積極：可接受較大損失，追求高報酬，請積極交易與參數建議。',
        'stable': '穩定：可接受小幅損失，追求穩定獲利，請平衡風險與報酬。',
        'conservative': '保守：損失容忍度極低，資產盡量不要減少，請極度保守建議。'
    }.get(risk_profile, '穩定：可接受小幅損失，追求穩定獲利，請平衡風險與報酬。')
    # 根據風險屬性動態調整風控條件
    risk_control_rules = {
        'aggressive': [
            '- 單筆交易損失可容忍上限為總資金的5%。',
            '- 單筆資金分配上限為80%。',
            '- 回撤期間可適度持續交易，但仍需說明理由。'
        ],
        'stable': [
            '- 單筆交易損失上限為總資金的2%。',
            '- 單筆資金分配上限為50%。',
            '- 回撤期間應降低風險或暫停交易，除非有明確理由。'
        ],
        'conservative': [
            '- 單筆交易損失上限為總資金的1%。',
            '- 單筆資金分配上限為30%。',
            '- 回撤期間應優先暫停交易，僅在極有把握時才可進場。'
        ]
    }
    risk_control_text = '\n'.join(risk_control_rules.get(risk_profile, risk_control_rules['stable']))
    prompt = f"""
你是一位專業量化交易 LLM，請根據下列市場數據與你剛才的分析，產生結構化決策：

[風險屬性] {risk_profile} - {risk_profile_desc}

[動態風控規則]
{risk_control_text}

[之前30天的分析摘要]
{free_analysis_text}

[市場 regime 狀態] {regime}
Current Date: {current_date}
Price Data (last 5 days):
{price_data_str}
Market News/Sentiment: {news_sentiment_summary}
LSTM預測未來{lstm_predict_day}天的Signal: {lstm_signal} ,
Recent performance: Last Trade PNL: {last_trade_pnl:.2f}, Cumulative PNL: {cumulative_pnl:.2f}
Total trades executed so far: {total_trades}
Current win rate: {win_rate_str}
Available strategies: {', '.join(available_strategies)}
Optimizer best parameters: {optimizer_best_params}
目前是否可以做空: {short_sell}

請務必只回傳一行，且完全依照下列格式，禁止多餘說明或換行：
STRATEGY_NAME{{"param1": value1, ..., "trailing_stop_pct": 0.1}}, expected_trades_per_month: X, capital_allocation_factor: Y (0.0~1.0), reason: <你的理由>, protection_action: <HALT_TRADING|REDUCE_RISK|RESUME_NORMAL|NONE>, protection_reason: <你的理由>

- 請根據市場波動、風險屬性，建議本次交易的 trailing_stop_pct（移動停損百分比，建議範圍 0.01~0.2），若無法判斷請回傳預設值。
請務必考慮上述動態風控規則、資金分配、參數動態調整，並說明你的信心與風險評估。
- 若你建議採用積極參數，必須提供 regime、績效或量價依據來佐證。
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), prompt
    except Exception as e:
        print(f"Error calling LLM for structured decision: {e}")
        return "", prompt
