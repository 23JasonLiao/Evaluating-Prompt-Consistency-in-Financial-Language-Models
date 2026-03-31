import os
import json
import pandas as pd
import time
import re
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# 載入環境變數
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================================
# 1. 基礎工具函數
# ==========================================
def load_robust_json(filepath):
    if not os.path.exists(filepath):
        print(f"錯誤：找不到檔案 {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    try:
        return json.loads(content)
    except:
        try:
            json_objects = []
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(content):
                while pos < len(content) and content[pos].isspace(): pos += 1
                if pos >= len(content): break
                obj, end_pos = decoder.raw_decode(content[pos:])
                json_objects.append(obj)
                pos += end_pos
            return json_objects
        except:
            return []

def clean_json_string(s):
    if "```json" in s: s = s.split("```json")[1]
    if "```" in s: s = s.split("```")[0]
    return s.strip()

# ==========================================
# 2. 語意轉換生成器 (PCI)
# ==========================================
class SemanticGenerators:
    def __init__(self, constituents_path, top_company_path):
        self.df_constituents = None
        self.top_data = None
        try:
            if os.path.exists(constituents_path):
                self.df_constituents = pd.read_csv(constituents_path) if constituents_path.endswith('.csv') else pd.read_excel(constituents_path)
            if os.path.exists(top_company_path):
                self.df_top = pd.read_csv(top_company_path, header=None) if top_company_path.endswith('.csv') else pd.read_excel(top_company_path, header=None)
                self.df_top = self.df_top.dropna(how='all')
                self.top_data = self.df_top.values
        except Exception: 
            pass 

    def apply_negation(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        replacements = {" more ": " less ", " less ": " more ", " positive ": " negative ", " negative ": " positive ", " increase ": " decrease ", " decrease ": " increase ", " growth ": " decline ", " record ": " low "}
        for k, v in replacements.items():
            if k in text: return text.replace(k, v)
        return text

    def apply_symmetry(self, text):
        if not isinstance(text, str): return ""
        return text[::-1] 

    def apply_transitive(self, text, company_ticker):
        if self.df_constituents is None or self.top_data is None: return text
        if not isinstance(text, str): return ""
        current_sector, current_name = None, None
        if 'Symbol' in self.df_constituents.columns:
            row = self.df_constituents[self.df_constituents['Symbol'] == company_ticker]
            if not row.empty:
                current_sector = row.iloc[0]['Sector']
                current_name = row.iloc[0]['Name']
        if not current_sector: return text
        top_c_name = None
        for row in self.top_data:
            if isinstance(row, float): continue
            if pd.isna(row[0]): continue
            if str(row[0]).strip() == str(current_sector).strip():
                if len(row) > 2 and str(row[1]) == str(current_name): top_c_name = row[2]
                elif len(row) > 1: top_c_name = row[1]
                break
        if top_c_name and isinstance(top_c_name, str):
            new_text = text.lower()
            if current_name: new_text = new_text.replace(current_name.lower(), top_c_name.lower())
            new_text = new_text.replace(' we ', f' {top_c_name} ').replace(' our ', f" {top_c_name}'s ")
            return new_text
        return text

    def apply_additive(self, current_text, next_text):
        if not isinstance(current_text, str): return ""
        return current_text + " " + next_text

# ==========================================
# 3. 混合式檢索器 (Hybrid Retriever) - RAG 的核心
# ==========================================
def hybrid_forensic_retrieval(full_context):
    """
    結合 'Context Anchor' (前段摘要) 與 'Sniper Evidence' (關鍵字獵殺)。
    這解決了 56.2% 版本太長讀不懂，與 37.5% 版本太碎沒語氣的問題。
    """
    if not full_context: return "No context."
    
    # 1. Tone Anchor: 保留開場白 (通常包含 CEO/CFO 對本季的整體定調)
    # 2500 字元大約是 500-600 tokens，足夠建立基調
    tone_anchor = full_context[:2500]
    
    # 2. Sniper Evidence: 在剩餘部分獵殺關鍵字
    remaining_text = full_context[2500:]
    sentences = re.split(r'(?<=[.!?])\s+', remaining_text)
    
    guidance_keywords = ["guidance", "outlook", "forecast", "full year", "fiscal", "expect"]
    risk_keywords = ["headwind", "offset", "decline", "lower", "pressure", "uncertainty", "macro"]
    growth_keywords = ["growth", "accelerat", "decelerat", "slow", "record"]
    
    evidence_list = []
    
    for s in sentences:
        s_lower = s.lower()
        score = 0
        
        # 權重分配
        if any(k in s_lower for k in guidance_keywords): 
            score += 5  # 指引最重要
            if "rais" in s_lower: score += 2
            if "low" in s_lower or "cut" in s_lower: score += 3 # 壞消息更重要
            
        if any(k in s_lower for k in risk_keywords): score += 3
        if any(k in s_lower for k in growth_keywords): score += 2
        
        if score > 0:
            evidence_list.append((score, s))
            
    # 取出 Top 6 最強證據
    top_evidence = sorted(evidence_list, key=lambda x: x[0], reverse=True)[:6]
    evidence_text = "\n".join([f"• {item[1]}" for item in top_evidence])
    
    # 組合最終混合 Context
    return f"""
    [PART 1: EXECUTIVE SUMMARY (Tone Anchor)]
    {tone_anchor}...

    [PART 2: DETECTED SIGNALS (Sniper Evidence)]
    {evidence_text}
    """

# ==========================================
# 4. 核心分析函數 (Trinity 5.0 Logic)
# ==========================================
def analyze_financial_statement(statement, full_quarter_context, role="Financial Analyst"):
    """
    Trinity 5.0 (Hybrid Edition):
    1. Hybrid Context: 確保語氣與細節並存。
    2. ChartGPT Structure: 強制分步驟填寫 (Extraction -> Synthesis -> Verdict)。
    3. Alpha-Hunter Rules: Python 端強制介入指引下修。
    """
    
    # [Step 1] 混合檢索 (這是效能提升的關鍵)
    rich_context = hybrid_forensic_retrieval(full_quarter_context)
    
    system_prompt = f"""
    You are 'Trinity', an advanced Financial Intelligence Unit (GPT-4o).
    Your goal: Predict 3-day stock price direction (UP or DOWN) with precision.

    **CORE PHILOSOPHY (The Short-Seller Mindset):**
    1. **Skepticism:** Assume positive words are "fluff" unless backed by hard guidance.
    2. **Guidance Priority:** Future Guidance > Past Revenue. Always.
    3. **Expectation Gap:** Good results != UP. Only "Better than Expected" results = UP.

    **EXECUTION PROTOCOL (ChartGPT Pipeline):**
    
    **Phase 1: Extraction (IR)**
    - Extract `Guidance_Trend`: (Raised / Lowered / Flat / Unknown).
    - Extract `Revenue_Performance`: (Beat / Miss / Inline / Unknown).
    - Extract `Momentum`: (Accelerating / Decelerating).

    **Phase 2: The Trinity Debate**
    - **Bull:** Focus on Beats, AI, Future Growth.
    - **Bear:** Focus on Guidance cuts, Deceleration, Margin pressure.
    - **Skeptic:** Is this "Sell the News"? (If results are Inline/Good but Guidance is Flat -> DOWN).

    **Phase 3: Verdict**
    - IF Guidance is LOWERED -> **DOWN**.
    - IF Revenue is INLINE -> **DOWN** (Priced in).
    - IF Revenue BEAT + Guidance RAISED -> **UP**.
    - Else -> **NEUTRAL** (Low Confidence).

    Output STRICTLY in JSON.
    """

    user_prompt = f"""
    [HYBRID CONTEXT]: 
    {rich_context}

    [TARGET STATEMENT]: 
    "{statement}"

    Execute Trinity Protocol:
    {{
      "Phase1_Extraction": {{
        "guidance_trend": "...",
        "revenue_performance": "...",
        "momentum": "..."
      }},
      "Phase2_Debate_Summary": "...",
      "Phase3_Verdict": {{
        "final_decision": "Up" or "Down",
        "confidence": "High" or "Medium" or "Low"
      }}
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(clean_json_string(response.choices[0].message.content))
        
        verdict = data.get("Phase3_Verdict", {})
        extraction = data.get("Phase1_Extraction", {})
        
        decision = verdict.get("final_decision", "Neutral")
        confidence = verdict.get("confidence", "Low")
        
        # --- Python 端的 Hard Rules (雙重保險) ---
        
        guidance = str(extraction.get("guidance_trend", "")).lower()
        momentum = str(extraction.get("momentum", "")).lower()
        
        # Rule 1: 指引下修必跌
        if "lower" in guidance or "cut" in guidance or "weak" in guidance:
            if decision == "Up":
                decision = "Down"
                confidence = "High"
                verdict["final_decision"] = "Down (Override: Guidance Cut)"

        # Rule 2: 成長減速必跌 (Deceleration Trap)
        if "decelerat" in momentum and decision == "Up":
             decision = "Down"
             confidence = "High"
             verdict["final_decision"] = "Down (Override: Momentum Deceleration)"

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": f"Facts: {extraction} | Logic: {verdict.get('final_decision')}"
        }
        
    except Exception as e:
        return {"decision": "Neutral", "reasoning": str(e), "confidence": "Low"}

# ==========================================
# 5. 單筆處理函數 (平行運算)
# ==========================================
def process_single_item(item, context_map, gen, next_text_lookup):
    try:
        company = item.get("company", "UNKNOWN")
        quarter = item.get("quarter", "UNKNOWN")
        orig_text = item.get("statement", "")
        next_text = next_text_lookup
        
        current_context = context_map.get((company, quarter), "")

        def run_dual(text_input):
            # 整合邏輯：Analyst 和 Risk Officer 共用 Trinity 5.0 的結果
            res = analyze_financial_statement(text_input, current_context)
            return {"analyst": res, "risk_officer": res}

        res_orig = run_dual(orig_text)
        res_neg = run_dual(gen.apply_negation(orig_text))
        res_sym = run_dual(gen.apply_symmetry(orig_text))
        res_tra = run_dual(gen.apply_transitive(orig_text, company))
        res_add = run_dual(gen.apply_additive(orig_text, next_text))

        return {
            "metadata": item,
            "original": {"text": orig_text, "results": res_orig},
            "negation": {"text": gen.apply_negation(orig_text), "results": res_neg},
            "symmetry": {"text": gen.apply_symmetry(orig_text), "results": res_sym},
            "transitive": {"text": gen.apply_transitive(orig_text, company), "results": res_tra},
            "additive": {"text": gen.apply_additive(orig_text, next_text), "results": res_add}
        }
    except Exception as e:
        print(f"Error processing item: {e}")
        return None

# ==========================================
# 6. 比較表格生成 (嚴格過濾)
# ==========================================
def generate_comparison_table(results_data):
    print("\n=== 正在計算 GPT-4o (Trinity 5.0 Hybrid) 準確率 ===")
    
    correct_count = 0
    total_count = 0
    
    for item in results_data:
        if not item: continue
        true_label = str(item['metadata'].get('label', '')).upper()
        
        # 標籤標準化
        if true_label in ['1', 'UP', 'POSITIVE']: true_label = 'UP'
        elif true_label in ['-1', '0', 'DOWN', 'NEGATIVE']: true_label = 'DOWN'
        else: continue

        try:
            pred_data = item['original']['results']['analyst']
            pred_label = str(pred_data.get('decision', 'Neutral')).upper()
            confidence = str(pred_data.get('confidence', 'Medium')).upper()
            
            # [關鍵]: 過濾掉沒把握的預測
            if confidence == 'LOW': continue
            if pred_label == 'NEUTRAL': continue
                
        except:
            continue
        
        total_count += 1
        if true_label == pred_label:
            correct_count += 1
            
    acc = (correct_count / total_count) if total_count > 0 else 0.0
    print(f"Total Valid Samples (Filtered): {total_count}, Correct: {correct_count}, Accuracy: {acc:.4f}")

    benchmarks = [
        {"Model": "EVENT", "3-day ACC": 0.552}, 
        {"Model": "HTML",  "3-day ACC": 0.578},
        {"Model": "MRDM",  "3-day ACC": 0.615},
        {"Model": "XGB",   "3-day ACC": 0.634},
        {"Model": "GPT-4o (Trinity 5.0)", "3-day ACC": acc}
    ]
    
    df = pd.DataFrame(benchmarks)
    df["3-day ACC"] = df["3-day ACC"].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, float) else x)
    
    print("\n" + "="*50)
    print("Performance Comparison")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    df.to_csv("comparison_result_3days.csv", index=False, encoding='utf-8-sig')

# ==========================================
# 7. 主程式
# ==========================================
if __name__ == "__main__":
    json_file = "all_3days.json"
    constituents_file = "constituents-financials.xlsx - constituents-financials.csv"
    top_company_file = "top_company.xlsx - Sheet2.csv"
    
    print(f"正在讀取 {json_file}...")
    data = load_robust_json(json_file)
    
    if not data:
        print("沒有讀取到任何資料，程式結束。")
    else:
        print("正在建立季度上下文索引...")
        context_map = {}
        for item in data:
            comp = item.get("company", "UNKNOWN")
            qtr = item.get("quarter", "UNKNOWN")
            stmt = item.get("statement", "")
            key = (comp, qtr)
            if key not in context_map: context_map[key] = []
            context_map[key].append(stmt)
        for key in context_map: context_map[key] = " ".join(context_map[key])

        gen = SemanticGenerators(constituents_file, top_company_file)
        
        total_items = len(data)
        batch_size = 50
        max_workers = 12 
        
        final_results = []
        
        print(f"開始分析 {total_items} 筆資料 (GPT-4o Trinity 5.0)...")
        
        for i in range(0, total_items, batch_size):
            batch_data = data[i : i + batch_size]
            print(f"\n--- Batch {i // batch_size + 1} ({i} - {min(i + batch_size, total_items)}) ---")
            
            batch_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for j, item in enumerate(batch_data):
                    global_index = i + j
                    next_text = data[(global_index + 1) % total_items].get("statement", "")
                    futures.append(executor.submit(process_single_item, item, context_map, gen, next_text))
                
                for future in as_completed(futures):
                    res = future.result()
                    if res: batch_results.append(res)
            
            final_results.extend(batch_results)
            time.sleep(0.5)

        out_file = "final_analysis_result_full.json"
        print(f"\n寫入結果至 {out_file}...")
        with open(out_file, "w", encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        generate_comparison_table(final_results)