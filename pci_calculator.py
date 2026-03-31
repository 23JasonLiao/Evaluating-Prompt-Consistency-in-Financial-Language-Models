import json
import pandas as pd
import os

def calculate_pci():
    # 1. 讀取分析結果
    input_file = "final_analysis_result_full.json"
    if not os.path.exists(input_file):
        print(f"錯誤：找不到 {input_file}，請先執行 main_analysis.py")
        return

    print(f"正在讀取 {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    
    # 用於統計 Risk Officer 的輔助檢查成效
    risk_audit_stats = {
        "total_checks": 0,
        "risk_validated": 0,
        "risk_flagged": 0
    }

    # 2. 定義輔助函數：提取指定角色的決策
    def get_decision(stage_data, role='analyst'):
        """
        從 data['original'] 等字典中提取指定角色 (analyst/risk_officer) 的決策
        """
        try:
            # 支援新版結構： results -> role -> decision
            if 'results' in stage_data and role in stage_data['results']:
                return stage_data['results'][role].get('decision', 'Neutral')
            
            # 相容舊版結構 (預設為 Analyst)
            if role == 'analyst' and 'analysis' in stage_data:
                return stage_data['analysis'].get('decision', 'Neutral')
                
            return "Neutral"
        except Exception:
            return "Neutral"

    # 3. 定義一致性檢查邏輯 (PCI 核心算法)
    def check_consistency(test_type, orig_dec, new_dec):
        """
        根據測試類型判斷是否一致 (Consistent)
        return: 100 (一致), 0 (不一致)
        """
        orig_dec = str(orig_dec).lower()
        new_dec = str(new_dec).lower()
        
        # 將文字轉為數值訊號
        def to_score(d):
            if "up" in d: return 1
            if "down" in d: return -1
            return 0

        orig_val = to_score(orig_dec)
        new_val = to_score(new_dec)

        # A. 否定測試 (Negation): 預期結果應「翻轉」
        if test_type == "negation":
            # Up -> Down 或 Down -> Up 為一致
            if orig_val * new_val == -1: return 100
            # Neutral -> Neutral 亦視為穩健
            if orig_val == 0 and new_val == 0: return 100
            return 0 

        # B. 其他測試 (Symmetry, Transitive, Additive): 預期結果應「相同」
        else:
            if orig_val == new_val: return 100
            return 0

    # 4. 迴圈計算每一筆資料
    for i, item in enumerate(data):
        quarter = item.get("metadata", {}).get("quarter", f"Item_{i+1}")
        
        # --- [Primary Role] Analyst: 負責生成主要訊號 ---
        orig_dec = get_decision(item['original'], role='analyst')
        neg_dec = get_decision(item['negation'], role='analyst')
        sym_dec = get_decision(item['symmetry'], role='analyst')
        tra_dec = get_decision(item['transitive'], role='analyst')
        add_dec = get_decision(item['additive'], role='analyst')

        # --- [Auxiliary Role] Risk Officer: 負責內部對抗性檢查 ---
        # 雖然不畫在圖上，但我們計算它與 Analyst 的一致性，作為信心參考
        risk_dec = get_decision(item['original'], role='risk_officer')
        
        # 執行 Risk Audit (幕後檢查)
        risk_audit_stats["total_checks"] += 1
        if str(orig_dec).lower() == str(risk_dec).lower():
            risk_audit_stats["risk_validated"] += 1
        else:
            risk_audit_stats["risk_flagged"] += 1

        # --- 計算 Analyst 的 PCI (用於視覺化) ---
        is_cons_orig = 100 
        is_cons_neg = check_consistency("negation", orig_dec, neg_dec)
        is_cons_sym = check_consistency("symmetry", orig_dec, sym_dec)
        is_cons_tra = check_consistency("transitive", orig_dec, tra_dec)
        is_cons_add = check_consistency("additive", orig_dec, add_dec)

        results.append({
            "Quarter": quarter,
            "Original": is_cons_orig,
            "Negation": is_cons_neg,
            "Symmetry": is_cons_sym,
            "Transitive": is_cons_tra,
            "Additive": is_cons_add
        })

    # 5. 彙總與輸出
    if not results:
        print("警告：沒有計算出任何結果。")
        return

    df = pd.DataFrame(results)
    pci_summary = df.groupby("Quarter").mean().reset_index()

    # 輸出給繪圖程式用的 CSV (只包含 Analyst 數據，保持圖表乾淨)
    output_csv = "pci_summary_result.csv"
    pci_summary.to_csv(output_csv, index=False, encoding='utf-8')
    
    # --- 顯示報告 ---
    print("\n=== PCI (Analyst Robustness) 計算完成 ===")
    print(pci_summary)
    
    print("\n=== [Internal Audit] Risk Officer Report ===")
    print(f"說明: Risk Officer 作為輔助角色，用於檢查 Analyst 訊號的潛在風險。")
    print(f"Total Transactions Audited: {risk_audit_stats['total_checks']}")
    print(f"Risk Validation Rate (Agreement): {risk_audit_stats['risk_validated'] / risk_audit_stats['total_checks']:.1%}")
    print(f"Risk Flags Raised (Disagreement): {risk_audit_stats['risk_flagged']}")
    print("-" * 40)
    
    print(f"\n視覺化數據已儲存至: {output_csv}")
    print("提示: 執行 drawcombined.py 時將只繪製 Analyst 的 PCI 雷達圖 (符合 Less is More 原則)。")

if __name__ == "__main__":
    calculate_pci()