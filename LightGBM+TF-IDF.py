import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import warnings
import joblib

files = [
    'Telecom_Fraud_Texts_5-main/label00-last.csv', 'Telecom_Fraud_Texts_5-main/label01-last.csv',
    'Telecom_Fraud_Texts_5-main/label02-last.csv', 'Telecom_Fraud_Texts_5-main/label03-last.csv',
    'Telecom_Fraud_Texts_5-main/label04-last.csv'
]

train_dfs, test_dfs = [], []

print("正在加载并切分数据 (前90%训练, 后10%测试)...")
for f in files:
    df = None
    # 确定性暴力尝试：先试标准 utf-8，失败则直接用最全的中文国标 gb18030
    for enc in ['utf-8', 'gb18030', 'gbk']:
        try:
            # 尝试按指定编码读取
            df = pd.read_csv(f, encoding=enc)
            print(f"✅ 文件 {f} 使用编码 [{enc}] 成功读取")
            break  # 读取成功就跳出编码尝试循环
        except UnicodeDecodeError:
            # 编码不对，静默进入下一个编码尝试
            continue
        except Exception as e:
            # 文件不存在或其他问题
            print(f"❌ 读取文件 {f} 时发生其他错误: {e}")
            break

    if df is None:
        print(f"⚠️ 无法读取文件 {f}，所有常用编码尝试均失败！")
        continue

    # 数据清洗
    df = df.dropna(subset=['content', 'label'])
    df['content'] = df['content'].astype(str)
    df['label'] = df['label'].astype(str).str.strip()

    # 切分验证集
    n_test = int(len(df) * 0.1)
    train_dfs.append(df.iloc[:-n_test].copy())
    test_dfs.append(df.iloc[-n_test:].copy())


train_data = pd.concat(train_dfs, ignore_index=True)
test_data = pd.concat(test_dfs, ignore_index=True)

# 2. TF-IDF 特征提取 (字符级 1-3 gram)
print("正在提取字符级TF-IDF特征...")
tfidf = TfidfVectorizer(max_features=15000, analyzer='char', ngram_range=(1, 3))
X_train = tfidf.fit_transform(train_data['content'])
X_test = tfidf.transform(test_data['content'])

# ==========================================
# 阶段一：二分类模型 (正常 vs 诈骗)
# ==========================================
print("\n=== 第一阶段：训练二分类模型 ===")
y_train_bin = (train_data['label'] != '0').astype(int)
y_test_bin = (test_data['label'] != '0').astype(int)

# 【代价敏感学习】: 给0(正常)设置极高的样本权重，1(诈骗)为常规权重
sample_weight_train = y_train_bin.map({0: 5.0, 1: 1.0}).values

clf_stage1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=42)
clf_stage1.fit(X_train, y_train_bin, sample_weight=sample_weight_train)

# 预测诈骗的概率
y_pred_prob_stage1 = clf_stage1.predict_proba(X_test)[:, 1]

# 【调整决策阈值】: 默认0.5，为了极低误判率，提高门槛至0.85
threshold = 0.85
y_pred_bin = (y_pred_prob_stage1 >= threshold).astype(int)

print(f"设定决策阈值: {threshold}")
print("二分类混淆矩阵:\n", confusion_matrix(y_test_bin, y_pred_bin))

# ==========================================
# 阶段二：多分类模型 (具体诈骗类型)
# ==========================================
print("\n=== 第二阶段：训练诈骗类型多分类模型 ===")
train_fraud = train_data[train_data['label'] != '0'].copy()

fraud_labels = train_fraud['label'].unique()
label_to_id = {l: i for i, l in enumerate(fraud_labels)}
id_to_label = {i: l for i, l in enumerate(fraud_labels)}

y_train_multi = train_fraud['label'].map(label_to_id).values
X_train_multi = tfidf.transform(train_fraud['content'])

# 多分类不需要处理极端的样本不平衡，常规训练即可
clf_stage2 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
clf_stage2.fit(X_train_multi, y_train_multi)

# ==========================================
# 整体管道级联评估
# ==========================================
test_data['fine_label'] = test_data['label'].apply(lambda x: '正常' if x == '0' else x)

# 默认全部初始化为正常
y_pred_final = np.array(['正常'] * len(test_data), dtype=object)

# 找到第一阶段被判定为诈骗的索引
fraud_indices = np.where(y_pred_bin == 1)[0]

if len(fraud_indices) > 0:
    X_test_fraud = X_test[fraud_indices]
    stage2_preds = clf_stage2.predict(X_test_fraud)

    # 覆盖第一阶段认为是诈骗的具体类别
    for idx, pred_id in zip(fraud_indices, stage2_preds):
        y_pred_final[idx] = id_to_label[pred_id]

print("\n=== 整体两阶段级联模型最终评估 ===")
print(classification_report(test_data['fine_label'], y_pred_final, digits=4))

# 将结果写入本地文件
test_data['predicted_label'] = y_pred_final
test_data['stage1_fraud_prob'] = y_pred_prob_stage1
test_data[['content', 'fine_label', 'predicted_label', 'stage1_fraud_prob']].to_csv('LightGBM+TF-IDF/test_predictions.csv', index=False)
joblib.dump(tfidf, 'LightGBM+TF-IDF/tfidf_vectorizer.pkl')
joblib.dump(clf_stage1, 'LightGBM+TF-IDF/clf_stage1.pkl')
joblib.dump(clf_stage2, 'LightGBM+TF-IDF/clf_stage2.pkl')
joblib.dump(id_to_label, 'LightGBM+TF-IDF/id_to_label.pkl')

print("✅ 模型和特征提取器已成功保存至本地！")