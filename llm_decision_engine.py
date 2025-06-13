import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import ast # Added import

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

def select_strategy_and_params(current_date, price_data_str, news_sentiment_summary, available_strategies, lstm_signal,
                               days_since_last_trade=0, max_days_no_trade=0, is_forced_trade_scenario=False,
                               recommended_strategy_name=None, recommended_strategy_signal=None,
                               current_active_strategy_name=None, current_strategy_days_active=0, min_strategy_duration=0,
                               last_trade_pnl=0.0, cumulative_pnl=0.0): # Added PNL context
    """
    Selects a trading strategy and its parameters using an LLM.
    In a forced trade scenario, decides whether to force a trade.
    When not in a forced scenario, selects a strategy target if conditions are met.
    """
    llm_decision_context = {"prompt": "", "response": ""}

    if is_forced_trade_scenario:
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
Price Data (last 5 days):
{price_data_str}
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
            return decision, None, llm_decision_context
        except Exception as e:
            print(f"Error calling Gemini API for forced trade: {e}")
            return "AbstainForceTrade", None, llm_decision_context # Default to abstain on error
    else: # This is for selecting/revising a strategy TARGET
        strategy_list_str = ", ".join(available_strategies)
        
        # Constructing the prompt for selecting a new strategy target
        prompt_intro = "You are a professional quant trading LLM.\nYour task is to select a trading strategy and its parameters for the next period."
        performance_context = f"\\nRecent performance: Last Trade PNL: {last_trade_pnl:.2f}, Cumulative PNL: {cumulative_pnl:.2f}."

        if current_active_strategy_name and current_active_strategy_name != "None":
            prompt_intro += f"\\nThe current strategy target is '{current_active_strategy_name}', which has been active for {current_strategy_days_active} days."
            prompt_intro += f" The minimum duration for a strategy is {min_strategy_duration} days."
            prompt_intro += performance_context
            if current_strategy_days_active >= min_strategy_duration:
                prompt_intro += "\\nYou now have the option to:"
                prompt_intro += f"\\n1. Re-select '{current_active_strategy_name}' with potentially **adjusted parameters**."
                prompt_intro += "\\n2. Select a completely new strategy target from the available list."
                if last_trade_pnl < 0:
                    prompt_intro += "\\n**Consider adjusting parameters for '{current_active_strategy_name}' or choosing a new strategy, as the last trade was not profitable.**"
            else:
                prompt_intro += f"\\nThe current strategy has not yet met its minimum duration of {min_strategy_duration} days."
                if last_trade_pnl < 0:
                    prompt_intro += "\\n**Even though the minimum duration is not met, if you strongly believe the current parameters for '{current_active_strategy_name}' are flawed given the recent negative PNL, you may suggest re-selecting it with new parameters. Otherwise, re-select it with current or slightly tweaked parameters.**"
                else:
                    prompt_intro += "\\nYou should ideally re-select it, possibly with minor parameter tweaks if deemed necessary."
        else:
            prompt_intro += "\\nNo strategy target is currently active. Please select an initial strategy target."
            prompt_intro += performance_context

        # 新增：明確給出參數建議區間與訊號頻率要求
        prompt_intro += "\n\nAvailable strategies: " + strategy_list_str + "."
        prompt_intro += "\nFor each strategy, here are the recommended parameter ranges:"
        prompt_intro += "\n- TrendStrategy: rsi_low_entry 35~55, rsi_high_entry 50~70, allow_short true/false"
        prompt_intro += "\n- RangeStrategy: window 10~30, rsi_low_entry 35~55, rsi_high_entry 50~70, rsi_mid_exit 45~55, allow_short true/false"
        prompt_intro += "\n- BreakoutStrategy: window 10~30, rsi_low 35~55, rsi_high 50~70, breakout_tol 0.98~1.02"
        prompt_intro += "\n- VolumePriceStrategy: volume_ratio 1.1~2.0, rsi_low 35~55, rsi_high 50~70"
        prompt_intro += "\n\n**Additional requirements:**"
        prompt_intro += "\n- The selected strategy and parameters should generate at least 1 trade signal per 10 trading days on average (based on recent data)."
        prompt_intro += "\n- If the last trade PNL is negative, you may consider switching strategy or adjusting parameters."
        prompt_intro += "\n- Avoid switching strategies too frequently; prefer parameter adjustment unless performance is poor."
        prompt_intro += "\n- **Do NOT always select the same strategy. You must dynamically choose the most suitable strategy based on recent market conditions, and avoid picking the same strategy for many consecutive periods.**"
        prompt_intro += "\n- Always provide a brief reason for your choice, especially if you keep the same strategy."
        prompt_intro += "\n- Return your answer in the format: STRATEGY_NAME{\"param1\": value1, \"param2\": value2, ...}, expected_trades_per_month: X, reason: <your_reason>"
        prompt_intro += "\n\nExample:"
        prompt_intro += "\nBreakoutStrategy{\"window\": 15, \"rsi_low\": 40, \"rsi_high\": 65, \"breakout_tol\": 0.995}, expected_trades_per_month: 4, reason: Market is consolidating and volatility is low."
        prompt_intro += "\n\nYour turn:"

        prompt = f"{prompt_intro}\nCurrent Date: {current_date}\nPrice Data (last 5 days):\n{price_data_str}\nMarket News/Sentiment (summarized): {news_sentiment_summary}\nLSTM Signal: {'Buy' if lstm_signal == 1 else 'Short' if lstm_signal == -1 else 'Neutral'}\n"
        llm_decision_context["prompt"] = prompt
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            llm_decision_context["response"] = response_text

            strategy_name = response_text.strip() 
            params = None 

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
                                print(f"Successfully parsed with json.loads: {parsed_dict} from '{s_to_parse}'")
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
                                    print(f"Successfully parsed with ast.literal_eval: {parsed_dict} from '{s_to_parse}'")
                                    break 
                                else:
                                    if s_idx == len(candidates_to_try) - 1: 
                                        print(f"ast.literal_eval did not return a dict for '{s_to_parse}': {type(evaluated_result)}")
                            except (ValueError, SyntaxError, TypeError): 
                                if s_idx == len(candidates_to_try) - 1: 
                                     print(f"Failed to parse '{s_to_parse}' with ast.literal_eval.")
                    
                    if parsed_dict is not None:
                        params = parsed_dict
                        strategy_name = name_part
                    else:
                        print(f"Could not parse parameters from '{actual_json_str}' after trying all methods. No params will be used.")
                        strategy_name = name_part 
                        params = None
                else: 
                    print(f"LLM response had malformed JSON part (no matching closing brace for initial '{{'): {response_text}. Treating as no params.")
                    strategy_name = response_text.split("{", 1)[0].strip() if "{" in response_text else response_text
                    params = None
            else: 
                strategy_name = response_text.strip()
                params = None

            # If LLM fails to select a valid strategy from the list, default to "Abstain"
            # main.py will interpret "Abstain" in this context as "LLM failed to pick a new target, continue with old or do nothing"
            if strategy_name not in available_strategies:
                print(f"LLM selected strategy '{strategy_name}' is not in available_strategies list: {available_strategies}. Defaulting to 'Abstain' (no change/no new target).")
                # Attempt to find if a known strategy is a substring, to catch minor LLM formatting issues
                found_known_strategy = False
                for known_strat in available_strategies:
                    if known_strat in strategy_name:
                        strategy_name = known_strat
                        print(f"Interpreted as known strategy: {strategy_name}")
                        found_known_strategy = True
                        break
                if not found_known_strategy:
                    strategy_name = "Abstain" # This signals main.py to not change the strategy or not set one if none active
                params = None # Ensure params are None if strategy is unknown/defaulted
            
            return strategy_name, params, llm_decision_context

        except Exception as e:
            print(f"Error calling Gemini API or parsing response for strategy selection: {e}")
            return "Abstain", None, llm_decision_context # Default to Abstain on error
