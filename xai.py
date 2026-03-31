import json
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib.font_manager import FontProperties

# 設定中文字型 (若有顯示中文需求，可選用系統內建字型)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

def perform_xai_analysis(json_file_path):
    print("1. 載入 LLM 分析結果...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"找不到檔案: {json_file_path}，請先執行 main_analysis.py")
        return

    # --- 資料準備 ---
    # 我們將提取「原始文本」作為特徵(X)，「分析師決策」作為目標(Y)
    corpus = []
    labels = []
    
    print(f"   共載入 {len(data)} 筆資料。")

    for entry in data:
        # 提取原始文本
        text = entry.get('original', {}).get('text', '')
        
        # 提取 LLM 的判斷 (Up/Down/Neutral)
        # 注意：這裡假設我們要解釋 'analyst' 的決策
        decision = entry.get('original', {}).get('results', {}).get('analyst', {}).get('decision', 'Neutral')
        
        if text and decision:
            corpus.append(text)
            labels.append(decision)

    if len(corpus) < 5:
        print("警告：資料量太少，XGBoost/SHAP 分析效果可能不佳。建議至少累積 50 筆以上資料。")

    df = pd.DataFrame({'text': corpus, 'label': labels})
    
    # --- 特徵工程 (NLP -> 數值) ---
    print("2. 進行文本向量化 (TF-IDF)...")
    # 使用 TF-IDF 找出關鍵詞權重，忽略太常見的停用詞
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    feature_names = vectorizer.get_feature_names_out()

    # --- 標籤編碼 ---
    # 將 Up/Down/Neutral 轉為 0, 1, 2
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    print(f"   標籤對應: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # --- 訓練 XGBoost 模型 ---
    print("3. 訓練 XGBoost 代理模型...")
    # 這裡我們用 XGBoost 來「模仿」LLM 的判斷邏輯
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # 為了演示，我們使用全部資料訓練 (實務上應分 Train/Test)
    model.fit(X, y)

    # --- SHAP 解釋 ---
    print("4. 計算 SHAP 值 (XAI)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- 視覺化 ---
    print("5. 產生解釋圖表...")
    
    # 若是多類別 (Up/Down/Neutral)，shap_values 會是一個 list，對應每個類別
    # 我們假設使用者最關心 "Up" (通常編碼後順序會變，需確認)
    # 下面嘗試找出 'Up' 對應的 index
    try:
        target_class_index = list(le.classes_).index('Up')
        print(f"   針對類別 'Up' 進行解釋分析...")
    except ValueError:
        # 如果沒有 Up，就取第一個類別
        target_class_index = 0
        print(f"   資料中無 'Up' 類別，針對 '{le.classes_[0]}' 進行分析...")

    # 處理 SHAP 值格式 (Binary vs Multiclass)
    if isinstance(shap_values, list):
        class_shap_values = shap_values[target_class_index]
    else:
        class_shap_values = shap_values

    # 1. Summary Plot (最直觀的圖：顯示哪些單字影響最大)
    plt.figure()
    shap.summary_plot(class_shap_values, X, feature_names=feature_names, show=False)
    plt.title(f"影響判斷為 '{le.classes_[target_class_index]}' 的關鍵字權重", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    print("   已儲存: shap_summary_plot.png")
    plt.close()

    # 2. Bar Plot (純粹的特徵重要性)
    plt.figure()
    shap.summary_plot(class_shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"Top 關鍵字重要性排行", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png')
    print("   已儲存: shap_bar_plot.png")
    plt.close()

    print("\n分析完成！請查看生成的 PNG 圖片以了解模型邏輯。")

if __name__ == "__main__":
    # 這裡指向 main_analysis.py 產出的結果檔
    input_json = "final_analysis_result_full.json"
    perform_xai_analysis(input_json)