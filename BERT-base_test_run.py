import os
# 为了保证国内网络依然能顺利加载基础网络结构 (MacBERT)，在引入transformers前加上 HF 镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. 仅加载测试集数据 (后 10%)
# ==========================================
def load_test_data(file_paths):
    test_dfs = []

    for idx, f in enumerate(file_paths):
        # 兼容 utf-8 和 gb18030
        try:
            df = pd.read_csv(f, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding='gb18030')
        except FileNotFoundError:
            print(f"文件不存在: {f}")
            continue

        n = len(df)
        split_idx = int(n * 0.9)

        # 只取后 10% 作为测试集
        test_df = df.iloc[split_idx:].copy()

        if 'label00' in f:
            test_df['stage1_label'] = 0
            test_df['stage2_label'] = -1
        else:
            test_df['stage1_label'] = 1
            test_df['stage2_label'] = idx - 1  # label01 -> 0, label02 -> 1...

        test_dfs.append(test_df)

    if not test_dfs:
        raise ValueError("没有找到任何测试数据文件")

    final_test_df = pd.concat(test_dfs, ignore_index=True)
    return final_test_df


# ==========================================
# 2. 核心评估流程
# ==========================================
def evaluate_model():
    file_paths = [
        'Telecom_Fraud_Texts_5-main/label00-last.csv',
        'Telecom_Fraud_Texts_5-main/label01-last.csv',
        'Telecom_Fraud_Texts_5-main/label02-last.csv',
        'Telecom_Fraud_Texts_5-main/label03-last.csv',
        'Telecom_Fraud_Texts_5-main/label04-last.csv'
    ]

    print("正在加载测试集数据 (后10%)...")
    test_df = load_test_data(file_paths)
    print(f"测试集样本总数: {len(test_df)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")

    # ==========================================
    # 3. 从本地加载分词器和模型权重
    # ==========================================
    MODEL_DIR = './BERT-base'
    base_model_name = 'hfl/chinese-macbert-base'  # 用于加载原始网络结构

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"找不到模型目录 {MODEL_DIR}，请确保训练已经完成并保存了模型。")

    print(f"\n正在从 {MODEL_DIR} 加载本地模型...")
    # 1. 加载本地的分词器
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

    # 2. 实例化两个基础网络结构
    from transformers import BertConfig
    config_s1 = BertConfig.from_pretrained(base_model_name, num_labels=2)
    model_s1 = BertForSequenceClassification(config_s1)
    
    config_s2 = BertConfig.from_pretrained(base_model_name, num_labels=4)
    model_s2 = BertForSequenceClassification(config_s2)

    # 3. 将我们训练好的 .pth 权重注入网络
    model_s1.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'stage1_weights.pth'), map_location=device))
    model_s2.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'stage2_weights.pth'), map_location=device))

    # 切换至评估模式并放入计算设备
    model_s1.eval()
    model_s2.eval()
    model_s1.to(device)
    model_s2.to(device)
    print("模型加载完毕！\n")

    # ==========================================
    # 4. 逐条推理并记录结果
    # ==========================================
    final_preds = []
    true_labels = []

    # 这里需要与你训练时的阈值保持一致
    THRESHOLD_SCAM = 0.85

    print("开始进行双阶段预测...")
    for idx, row in test_df.iterrows():
        text = str(row['content'])
        true_stage1 = row['stage1_label']
        true_stage2 = row['stage2_label']

        # 将真实标签映射为 0(正常), 1,2,3,4(具体诈骗类型)
        actual_class = 0 if true_stage1 == 0 else true_stage2 + 1
        true_labels.append(actual_class)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding='max_length'
        ).to(device)

        with torch.no_grad():
            # 阶段一预测
            outputs_s1 = model_s1(**inputs)
            probs_s1 = torch.nn.functional.softmax(outputs_s1.logits, dim=-1)
            prob_scam = probs_s1[0][1].item()  # 获取判定为诈骗(1)的概率

            # 如果达到阈值，进入阶段二预测具体类型
            if prob_scam >= THRESHOLD_SCAM:
                outputs_s2 = model_s2(**inputs)
                pred_s2 = torch.argmax(outputs_s2.logits, dim=-1).item()
                final_preds.append(pred_s2 + 1)  # 预测类别为 1~4
            else:
                final_preds.append(0)  # 否则视为正常短信

        # 进度打印
        if (idx + 1) % 200 == 0:
            print(f"已处理 {idx + 1}/{len(test_df)} 条测试样本")

    print("\n========== 测试集双阶段评估报告 ==========")
    target_names = ['正常(0)', '诈骗类型1(1)', '诈骗类型2(2)', '诈骗类型3(3)', '诈骗类型4(4)']
    print(classification_report(true_labels, final_preds, target_names=target_names, digits=4))

    accuracy = (np.array(true_labels) == np.array(final_preds)).mean()
    print(f"\n总体准确率 (Accuracy): {accuracy:.4f}")

    # 额外统计：误杀率（把正常判为诈骗）
    true_labels_arr = np.array(true_labels)
    final_preds_arr = np.array(final_preds)

    normal_idx = (true_labels_arr == 0)
    total_normal = normal_idx.sum()
    false_positive = (final_preds_arr[normal_idx] != 0).sum()

    print(f"正常短信总数: {total_normal}")
    print(f"正常短信被误判为诈骗的数量: {false_positive}")
    print(f"误杀率 (False Positive Rate): {false_positive / total_normal * 100:.2f}%\n")


if __name__ == "__main__":
    evaluate_model()