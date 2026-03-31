import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import numpy as np

# 設定學術風格繪圖參數
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def load_data(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return []

def calculate_metrics(data):
    """
    [自動化核心]
    從 JSON 真實數據即時計算 ACC, F1, MCC 與 PCI。
    不使用任何 Hardcoded 數值。
    """
    y_true = []
    y_pred = []
    
    # Consistency Counters
    cons_stats = {
        "Negation": {"match": 0, "total": 0},
        "Symmetry": {"match": 0, "total": 0},
        "Transitive": {"match": 0, "total": 0},
        "Additive": {"match": 0, "total": 0}
    }

    for item in data:
        # --- 1. 提取 Ground Truth ---
        raw_label = str(item.get("metadata", {}).get("label", "")).upper()
        if raw_label in ["UP", "1", "POSITIVE"]: 
            gt = 1
        elif raw_label in ["DOWN", "0", "-1", "NEGATIVE"]: 
            gt = 0
        else: 
            continue # 跳過無效標籤

        # --- 2. 提取 Prediction ---
        try:
            res_orig = item["original"]["results"]["analyst"]["decision"].upper()
            if res_orig == "UP":
                pred = 1
            elif res_orig == "DOWN":
                pred = 0
            else:
                continue # 跳過 Neutral 或格式錯誤
        except: 
            continue

        y_true.append(gt)
        y_pred.append(pred)

        # --- 3. 計算 Consistency (PCI) ---
        def get_dec(key):
            try: return item[key]["results"]["analyst"]["decision"].upper()
            except: return None

        d_orig = res_orig
        
        # (A) Negation
        d_neg = get_dec("negation")
        if d_neg in ["UP", "DOWN"]:
            cons_stats["Negation"]["total"] += 1
            if d_orig != d_neg: cons_stats["Negation"]["match"] += 1
            
        # (B) Symmetry
        d_sym = get_dec("symmetry")
        if d_sym in ["UP", "DOWN"]:
            cons_stats["Symmetry"]["total"] += 1
            if d_orig == d_sym: cons_stats["Symmetry"]["match"] += 1

        # (C) Transitive
        d_tra = get_dec("transitive")
        if d_tra in ["UP", "DOWN"]:
            cons_stats["Transitive"]["total"] += 1
            if d_orig == d_tra: cons_stats["Transitive"]["match"] += 1

        # (D) Additive
        d_add = get_dec("additive")
        if d_add in ["UP", "DOWN"]:
            cons_stats["Additive"]["total"] += 1
            if d_orig == d_add: cons_stats["Additive"]["match"] += 1

    # --- 4. 使用 sklearn 自動計算指標 ---
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro') # Macro F1 for balanced evaluation
        mcc = matthews_corrcoef(y_true, y_pred)
    else:
        acc, f1, mcc = 0, 0, 0
    
    # 計算 PCI 百分比
    pci = {}
    for k, v in cons_stats.items():
        pci[k] = (v["match"] / v["total"]) * 100 if v["total"] > 0 else 0

    return acc, f1, mcc, pci

def get_benchmark_data(gpt_acc, gpt_f1, gpt_mcc, gpt_pci):
    """
    建立完整的比較數據集。
    GPT-4o 數據來自自動計算，其他來自 FinGPT 論文基準。
    """
    
    # --- Chart 1: Performance (ACC, F1, MCC) ---
    perf_data = [
        # Model, Metric, Score
        ("EVENT", "Accuracy", 0.416), ("EVENT", "F1-Score", 0.582), ("EVENT", "MCC", 0.078),
        ("HTML",  "Accuracy", 0.442), ("HTML",  "F1-Score", 0.571), ("HTML",  "MCC", 0.052),
        ("MRDM",  "Accuracy", 0.504), ("MRDM",  "F1-Score", 0.541), ("MRDM",  "MCC", 0.079),
        ("XGB",   "Accuracy", 0.434), ("XGB",   "F1-Score", 0.448), ("XGB",   "MCC", -0.093),
        ("GPT-4o (Trinity)", "Accuracy", gpt_acc), 
        ("GPT-4o (Trinity)", "F1-Score", gpt_f1), 
        ("GPT-4o (Trinity)", "MCC", gpt_mcc)
    ]
    df_perf = pd.DataFrame(perf_data, columns=["Model", "Metric", "Score"])

    # --- Chart 3: Consistency (PCI across 4 variants) ---
    cons_data = []
    
    # 1. Negation
    cons_data.extend([
        ("EVENT", "Negation", 10.6), ("HTML", "Negation", 11.5), 
        ("MRDM", "Negation", 24.8),  ("XGB", "Negation", 7.1),
        ("GPT-4o (Trinity)", "Negation", gpt_pci["Negation"])
    ])
    
    # 2. Symmetry
    cons_data.extend([
        ("EVENT", "Symmetry", 94.7), ("HTML", "Symmetry", 89.4), 
        ("MRDM", "Symmetry", 65.5),  ("XGB", "Symmetry", 100),
        ("GPT-4o (Trinity)", "Symmetry", gpt_pci["Symmetry"])
    ])
    
    # 3. Transitive
    cons_data.extend([
        ("EVENT", "Transitive", 96.5), ("HTML", "Transitive", 89.4), 
        ("MRDM", "Transitive", 82.3),  ("XGB", "Transitive", 67.3),
        ("GPT-4o (Trinity)", "Transitive", gpt_pci["Transitive"])
    ])
    
    # 4. Additive
    cons_data.extend([
        ("EVENT", "Additive", 90.3), ("HTML", "Additive", 69.9), 
        ("MRDM", "Additive", 66.4),  ("XGB", "Additive", 52.2),
        ("GPT-4o (Trinity)", "Additive", gpt_pci["Additive"])
    ])
    
    df_cons = pd.DataFrame(cons_data, columns=["Model", "Variant", "Consistency (%)"])
    
    return df_perf, df_cons

def plot_charts(df_perf, df_cons):
    # --- Chart 1: Performance Comparison ---
    plt.figure(figsize=(12, 6))
    ax1 = sns.barplot(x="Metric", y="Score", hue="Model", data=df_perf, palette="viridis", edgecolor="white", linewidth=1)
    
    plt.title("Model Performance Comparison (3-Day Horizon)", fontsize=16, fontweight='bold', pad=20)
    plt.ylim(-0.15, 0.85) # 設定 Y 軸範圍 (包含負值空間)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("")
    
    # Add labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
    plt.tight_layout()
    plt.savefig("chart1_performance.png", dpi=300)
    print("Generated Chart 1: chart1_performance.png")

    # --- Chart 3: Consistency Comparison ---
    plt.figure(figsize=(12, 6))
    ax2 = sns.barplot(x="Variant", y="Consistency (%)", hue="Model", data=df_cons, palette="magma", edgecolor="white", linewidth=1)
    
    plt.title("Decision Consistency Rate (DCR)", fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 119) # 留空間給標籤
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylabel("Consistency Score (%)", fontsize=12)
    plt.xlabel("Transformation Type", fontsize=12)
    
    # Add labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig("chart3_consistency.png", dpi=300)
    print("Generated Chart 3: chart3_consistency.png")

def main():
    json_file = "final_analysis_result_full.json"
    data = load_data(json_file)
    
    if not data: 
        print("No data found!")
        return

    # 1. 自動計算指標
    acc, f1, mcc, pci = calculate_metrics(data)
    print(f"GPT-4o Auto-Calculated -> ACC: {acc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
    print(f"PCI -> {pci}")

    # 2. 整合基準數據
    df_perf, df_cons = get_benchmark_data(acc, f1, mcc, pci)

    # 3. 輸出 CSV
    df_perf.to_csv("comparison_result_3days_auto.csv", index=False)
    df_cons.to_csv("consistency_result_3days_auto.csv", index=False)

    # 4. 畫圖
    plot_charts(df_perf, df_cons)

if __name__ == "__main__":
    main()