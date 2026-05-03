import mindspore as ms
from mindspore import Tensor
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import os

# 强制将计算图设置为CPU，并演示数据在MindSpore Tensor与NumPy间的无缝转换
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

def load_and_preprocess_data(data_dir="Telecom_Fraud_Texts_5-main"):
    print("正在加载并切分数据 (前90%训练, 后10%测试)...")
    files = [
        'label00-last.csv', 'label01-last.csv',
        'label02-last.csv', 'label03-last.csv',
        'label04-last.csv'
    ]
    train_dfs, test_dfs = [], []
    for idx, f in enumerate(files):
        file_path = os.path.join(data_dir, f)
        df = None
        for enc in ['utf-8', 'gb18030', 'gbk']:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                break
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        if df is None:
            continue
            
        df = df.dropna(subset=['content'])
        df['content'] = df['content'].astype(str)
        # 标签映射：0为正常，1~4为各类诈骗
        df['label'] = idx
        
        n_test = int(len(df) * 0.1)
        train_dfs.append(df.iloc[:-n_test].copy())
        test_dfs.append(df.iloc[-n_test:].copy())

    if not train_dfs:
        return pd.DataFrame(), pd.DataFrame()

    train_data = pd.concat(train_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)
    return train_data, test_data

def main():
    train_data, test_data = load_and_preprocess_data()
    if len(train_data) == 0:
        print("未找到数据文件，请检查目录 Telecom_Fraud_Texts_5-main！")
        return
        
    print("正在提取字符级 TF-IDF 特征...")
    # 最大词表设为15000，考虑字符组合防止错别字
    tfidf = TfidfVectorizer(max_features=15000, analyzer='char', ngram_range=(1, 3))
    X_train = tfidf.fit_transform(train_data['content']).toarray()
    X_test = tfidf.transform(test_data['content']).toarray()
    
    # 演示：将特征转化为 MindSpore Tensor，这在与深度学习网络混合使用时极其重要
    ms_X_train = Tensor(X_train, dtype=ms.float32)
    ms_X_test = Tensor(X_test, dtype=ms.float32)
    
    # ==========================================
    # 阶段一：二分类模型 (正常 vs 诈骗)
    # ==========================================
    print("\n=== 第一阶段：训练二分类模型 ===")
    y_train_bin = (train_data['label'] != 0).astype(int)
    y_test_bin = (test_data['label'] != 0).astype(int)
    
    # 【代价敏感学习】: 给0(正常)设置权重为5.0，严格防误杀
    sample_weight = y_train_bin.map({0: 5.0, 1: 1.0}).values
    
    clf_stage1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=42)
    # 将Tensor转回NumPy供LightGBM训练
    clf_stage1.fit(ms_X_train.asnumpy(), y_train_bin, sample_weight=sample_weight)
    
    # 推理并设定阈值
    y_pred_prob_stage1 = clf_stage1.predict_proba(ms_X_test.asnumpy())[:, 1]
    THRESHOLD = 0.85
    y_pred_bin = (y_pred_prob_stage1 >= THRESHOLD).astype(int)
    print(f"阶段一判定阈值设定为 {THRESHOLD}")
    
    # ==========================================
    # 阶段二：具体诈骗类型多分类
    # ==========================================
    print("\n=== 第二阶段：训练具体诈骗类型模型 ===")
    fraud_mask = train_data['label'] != 0
    X_train_multi = ms_X_train.asnumpy()[fraud_mask]
    y_train_multi = train_data[fraud_mask]['label'].values
    
    clf_stage2 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
    clf_stage2.fit(X_train_multi, y_train_multi)
    
    # ==========================================
    # 整体管道级联评估
    # ==========================================
    print("\n========== 两阶段串联分类最终报告 ==========")
    final_preds = np.zeros(len(test_data), dtype=int)
    fraud_indices = np.where(y_pred_bin == 1)[0]
    
    if len(fraud_indices) > 0:
        X_test_fraud = ms_X_test.asnumpy()[fraud_indices]
        stage2_preds = clf_stage2.predict(X_test_fraud)
        final_preds[fraud_indices] = stage2_preds
        
    target_names = ['正常(0)', '诈骗类型1(1)', '诈骗类型2(2)', '诈骗类型3(3)', '诈骗类型4(4)']
    print(classification_report(test_data['label'], final_preds, target_names=target_names, digits=4))
    print("模型训练并评估完毕！")

if __name__ == "__main__":
    main()
