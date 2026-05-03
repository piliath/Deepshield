import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. 指定你的文件路径 (保持与训练时一致)
files = [
    'Telecom_Fraud_Texts_5-main/label00-last.csv',
    'Telecom_Fraud_Texts_5-main/label01-last.csv',
    'Telecom_Fraud_Texts_5-main/label02-last.csv',
    'Telecom_Fraud_Texts_5-main/label03-last.csv',
    'Telecom_Fraud_Texts_5-main/label04-last.csv'
]

test_dfs = []

print("正在加载后10%验证集数据...")
for f in files:
    df = None
    # 确定性暴力尝试：先试标准 utf-8，失败则直接用最全的中文国标 gb18030，再兜底 gbk
    for enc in ['utf-8', 'gb18030', 'gbk']:
        try:
            df = pd.read_csv(f, encoding=enc)
            print(f"✅ 文件 {f} 使用编码 [{enc}] 成功读取")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"❌ 读取文件 {f} 时发生其他错误: {e}")
            break

    if df is None:
        print(f"⚠️ 无法读取文件 {f}，所有常用编码尝试均失败！")
        continue

    df = df.dropna(subset=['content', 'label'])
    df['content'] = df['content'].astype(str)
    df['label'] = df['label'].astype(str).str.strip()

    # 严格按照训练时的逻辑：取每个文件的最后 10%
    n_test = int(len(df) * 0.1)
    test_dfs.append(df.iloc[-n_test:].copy())

test_data = pd.concat(test_dfs, ignore_index=True)
print(f"\n验证集提取完毕，共计: {len(test_data)} 条数据。\n")

# 2. 从本地加载所有训练好的模型组件
print("正在从本地加载 .pkl 模型文件...")
try:
    # 注意这里使用的是正斜杠 /
    tfidf = joblib.load('LightGBM+TF-IDF/tfidf_vectorizer.pkl')
    clf_stage1 = joblib.load('LightGBM+TF-IDF/clf_stage1.pkl')
    clf_stage2 = joblib.load('LightGBM+TF-IDF/clf_stage2.pkl')
    id_to_label = joblib.load('LightGBM+TF-IDF/id_to_label.pkl')
    print("✅ 模型加载成功！\n")
except FileNotFoundError as e:
    print(f"❌ 找不到模型文件，请确认它们和当前脚本在同一目录下。报错信息: {e}")
    exit()

# 3. 数据预处理与特征转换 (注意：这里只用 transform，绝对不能用 fit)
print("正在转换 TF-IDF 特征...")
X_test = tfidf.transform(test_data['content'])

# 4. 阶段一：二分类预测 (正常 vs 诈骗)
threshold = 0.85
y_pred_prob_stage1 = clf_stage1.predict_proba(X_test)[:, 1]
# 大于等于阈值判为诈骗(1)，否则为正常(0)
y_pred_bin = (y_pred_prob_stage1 >= threshold).astype(int)

# 5. 阶段二：多分类级联预测
# 初始化最终预测结果都为“正常”
y_pred_final = np.array(['正常'] * len(test_data), dtype=object)

# 找到被第一阶段判定为诈骗的样本索引
fraud_indices = np.where(y_pred_bin == 1)[0]

if len(fraud_indices) > 0:
    # 只把嫌疑短信送入阶段二多分类器
    X_test_fraud = X_test[fraud_indices]
    stage2_preds = clf_stage2.predict(X_test_fraud)

    # 将多分类器输出的ID映射回真实的中文标签，并覆盖最终结果
    for idx, pred_id in zip(fraud_indices, stage2_preds):
        y_pred_final[idx] = id_to_label[pred_id]

# 6. 生成评估报告
# 将测试集的 '0' 统一转换为 '正常' 以便直观比对
test_data['fine_label'] = test_data['label'].apply(lambda x: '正常' if x == '0' else x)

print("=== 验证集 (后10%) 最终分类报告 ===")
print(classification_report(test_data['fine_label'], y_pred_final, digits=4))

# 输出误报分析 (把正常短信误拦截的情况)
false_positives = test_data[(test_data['fine_label'] == '正常') & (y_pred_final != '正常')]
print(f"\n=========================================")
print(f"🎯 核心指标：把正常短信误判为诈骗的数量为 -> {len(false_positives)} 条")
print(f"=========================================")

if len(false_positives) > 0:
    print("\n[误判的正常短信样例]:")
    for _, row in false_positives.head(5).iterrows():
        print(f"内容: {row['content'][:60]}...")