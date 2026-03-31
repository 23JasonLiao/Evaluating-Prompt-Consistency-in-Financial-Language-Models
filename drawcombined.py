import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import base64
import json  # 新增: 用於讀取 JSON 檔案
from io import BytesIO
from math import pi
# 新增: 用於 t-SNE 和混淆矩陣的套件
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# --- 設定繪圖風格 ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

def fig_to_base64(fig):
    """將 matplotlib 圖表轉換為 HTML 可讀的 Base64 字串"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120) # 提高解析度
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# ==========================================
# 新增功能: 讀取 JSON 資料
# ==========================================
def load_json_data(filepath):
    if not os.path.exists(filepath):
        print(f"錯誤: 找不到檔案 {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==========================================
# 新增圖表 1: 混淆矩陣 (Confusion Matrix)
# ==========================================
def plot_confusion_matrix(data):
    print("正在生成圖表: 混淆矩陣...")
    y_true = []
    y_pred = []
    
    for item in data:
        true_label = str(item['metadata'].get('label', '')).upper()
        try:
            # 嘗試取得 analyst 的決策，若無則取外層的決策
            if 'analyst' in item['original']['results']:
                pred = item['original']['results']['analyst'].get('decision', 'Neutral')
            else:
                pred = item['original']['results'].get('decision', 'Neutral')
            pred_label = str(pred).upper()
        except:
            continue
            
        # 只統計有明確 UP/DOWN 的資料
        if true_label in ['UP', 'DOWN'] and pred_label in ['UP', 'DOWN']:
            y_true.append(true_label)
            y_pred.append(pred_label)
    
    if not y_true:
        print("警告: 沒有足夠的資料生成混淆矩陣。")
        return None

    labels = ['DOWN', 'UP']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": 14}, ax=ax)
    ax.set_title('Model Performance: Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return fig_to_base64(fig)

# ==========================================
# 新增圖表 2: t-SNE 分佈圖
# ==========================================
def plot_tsne(data):
    print("正在生成圖表: t-SNE 語意分佈...")
    corpus = []
    labels = []
    predictions = []
    
    for entry in data:
        text = entry.get('original', {}).get('text', '')
        true_label = str(entry['metadata'].get('label', '')).upper()
        try:
            if 'analyst' in entry['original']['results']:
                pred_label = str(entry['original']['results']['analyst']['decision']).upper()
            else:
                pred_label = str(entry['original']['results']['decision']).upper()
        except:
            pred_label = "NEUTRAL"

        if text and (true_label in ['UP', 'DOWN']):
            corpus.append(text)
            labels.append(true_label)
            predictions.append(pred_label)

    if len(corpus) < 5:
        print("警告: 資料過少，跳過 t-SNE。")
        return None

    # 1. 文本向量化 (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
    X = vectorizer.fit_transform(corpus).toarray()

    # 2. 降維 (PCA -> t-SNE)
    n_pca = min(50, len(corpus))
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=min(30, len(corpus)-1), random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_pca)

    # 3. 準備繪圖資料
    df_tsne = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'Label': labels, 'Prediction': predictions})
    df_tsne['Status'] = df_tsne.apply(lambda row: 'Correct' if row['Label'] == row['Prediction'] else 'Wrong', axis=1)

    # 4. 繪圖
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='x', y='y', hue='Label', style='Status', palette={'UP': '#e74c3c', 'DOWN': '#3498db'}, markers={'Correct': 'o', 'Wrong': 'X'}, s=100, alpha=0.8, ax=ax)
    ax.set_title('t-SNE Semantic Visualization', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig_to_base64(fig)

# ==========================================
# 原有程式碼 (保持不變)
# ==========================================
def create_radar_chart(df):
    """繪製雷達圖 (原本的 visualize_combined)"""
    categories = ['Original', 'Negation', 'Symmetry', 'Transitive', 'Additive']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    
    for i, row in df.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        
        quarter_label = row['Quarter']
        color = colors[i % len(colors)]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=quarter_label, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
    plt.ylim(0, 100)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('PCI Radar Chart: Consistency Across Dimensions', size=15, weight='bold', y=1.1)
    
    return fig_to_base64(fig)

def create_heatmap(df):
    """繪製熱圖 (原本的 visualize_combined)"""
    data = df.set_index('Quarter')[['Original', 'Negation', 'Symmetry', 'Transitive', 'Additive']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu", 
                vmin=0, vmax=100, linewidths=.5, ax=ax)
    
    ax.set_title('PCI Heatmap: Consistency Scores (%)', fontsize=14, fontweight='bold', pad=15)
    return fig_to_base64(fig)

def create_parallel_chart(df):
    """繪製平行座標圖 (原本的 visualize_parallel)"""
    from pandas.plotting import parallel_coordinates
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 為了畫圖美觀，只取需要的欄位
    cols = ['Quarter', 'Original', 'Negation', 'Symmetry', 'Transitive', 'Additive']
    plot_data = df[cols].copy()
    
    # 手動指定顏色
    colors = ['#4c72b0', '#c44e52', '#55a868', '#8172b3', '#ccb974']
    
    parallel_coordinates(plot_data, 'Quarter', color=colors, linewidth=2.5, alpha=0.8, ax=ax)
    
    plt.title('PCI Parallel Coordinates Profile', fontsize=16, fontweight='bold')
    plt.ylabel('Score (%)')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    return fig_to_base64(fig)

# ==========================================
# 更新 HTML 生成函數 (加入新圖表)
# ==========================================
def generate_html(cm_img, tsne_img, radar_img, heatmap_img, parallel_img):
    html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <title>Financial Analysis Combined Report</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-radius: 10px; }}
            h1 {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
            h2 {{ color: #2980b9; margin-top: 50px; border-left: 5px solid #2980b9; padding-left: 10px; }}
            .chart-container {{ text-align: center; margin: 30px 0; padding: 20px; background: #fff; border: 1px solid #eee; border-radius: 8px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 5px; }}
            .description {{ font-size: 0.95em; color: #666; margin-bottom: 15px; line-height: 1.6; text-align: left; background: #f9f9f9; padding: 15px; border-radius: 5px; }}
            .footer {{ text-align: center; margin-top: 50px; font-size: 0.8em; color: #999; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>全方位金融文本分析報告 (5 Charts)</h1>

            <div class="chart-container">
                <h2>1. 模型效能: 混淆矩陣 (Confusion Matrix)</h2>
                <p class="description">顯示模型預測漲跌的準確度。對角線顏色越深代表預測越準確，非對角線代表誤判。</p>
                <img src="data:image/png;base64,{cm_img}" />
            </div>

            <div class="chart-container">
                <h2>2. 語意分佈: t-SNE Visualization</h2>
                <p class="description">將高維文本特徵降維至 2D 平面。紅色為漲，藍色為跌。觀察紅藍點是否能被有效區分，以及錯誤預測點 (X) 的分佈位置。</p>
                <img src="data:image/png;base64,{tsne_img}" />
            </div>

            <div class="chart-container">
                <h2>3. 穩健性分析: PCI 雷達圖 (Radar Chart)</h2>
                <p class="description">此圖展示不同季度/公司在五個維度上的綜合得分。面積越大代表模型對該季度的判斷越穩健 (Robust)。</p>
                <img src="data:image/png;base64,{radar_img}" />
            </div>

            <div class="chart-container">
                <h2>4. 穩健性分析: PCI 熱圖 (Heatmap)</h2>
                <p class="description">熱圖以顏色深淺直觀顯示分數高低。深色代表高一致性，淺色代表模型容易受到語意擾動影響。</p>
                <img src="data:image/png;base64,{heatmap_img}" />
            </div>

            <div class="chart-container">
                <h2>5. 穩健性分析: PCI 平行座標圖 (Parallel Coordinates)</h2>
                <p class="description">此圖用於追蹤各個維度的變化趨勢，可以觀察到是否有特定的語意變換 (如 Negation) 導致所有季度的分數普遍下降。</p>
                <img src="data:image/png;base64,{parallel_img}" />
            </div>

            <div class="footer">
                Generated by Combined Analysis Pipeline
            </div>
        </div>
    </body>
    </html>
    """
    return html

# ==========================================
# 主程式
# ==========================================
def main():
    # 定義檔案路徑
    json_file = "final_analysis_result_full.json"
    csv_file = "pci_summary_result.csv"
    output_file = "analysis_report_combined.html"

    # 1. 讀取資料
    print(f"讀取 JSON 資料: {json_file} ...")
    json_data = load_json_data(json_file)
    
    print(f"讀取 CSV 資料: {csv_file} ...")
    if not os.path.exists(csv_file):
        print(f"錯誤：找不到 {csv_file}。請先執行 pci_calculator.py！")
        return
    df = pd.read_csv(csv_file)

    if not json_data:
        print("錯誤: JSON 資料讀取失敗，無法生成混淆矩陣與 t-SNE 圖。")
        return

    # 2. 生成新圖表 (使用 JSON 資料)
    print("正在生成新圖表...")
    cm_b64 = plot_confusion_matrix(json_data)
    tsne_b64 = plot_tsne(json_data)

    # 3. 生成原有圖表 (使用 CSV 資料)
    print("正在生成原有圖表...")
    radar_b64 = create_radar_chart(df)
    heatmap_b64 = create_heatmap(df)
    parallel_b64 = create_parallel_chart(df)

    # 4. 組裝 HTML
    if cm_b64 and tsne_b64 and radar_b64 and heatmap_b64 and parallel_b64:
        print("正在組裝 HTML 報告...")
        html_content = generate_html(cm_b64, tsne_b64, radar_b64, heatmap_b64, parallel_b64)
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"成功！報告已生成於: {output_file}")
    else:
        print("錯誤: 部分圖表生成失敗，無法建立完整報告。")

if __name__ == "__main__":
    main()