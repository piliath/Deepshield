import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


# ================= 1. 定义与保存时完全一致的模型结构 =================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=[2, 3, 4], num_filters=64):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        pooled_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            p = torch.max_pool1d(c, c.size(2)).squeeze(2)
            pooled_outputs.append(p)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool = self.dropout(h_pool)
        out = self.fc(h_pool)
        return out


# ================= 2. 加载本地模型与配置 =================
print("正在加载本地模型文件...")
model_dir = "TextCNN"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载配置和词表
with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
with open(os.path.join(model_dir, "vocab.json"), "r", encoding="utf-8") as f:
    vocab = json.load(f)

# 初始化并加载权重
model_1 = TextCNN(config["vocab_size"], config["embed_dim"], 2, config["filter_sizes"], config["num_filters"]).to(
    device)
model_1.load_state_dict(torch.load(os.path.join(model_dir, "model_1_stage1.pth"), map_location=device))
model_1.eval()

model_2 = TextCNN(config["vocab_size"], config["embed_dim"], 4, config["filter_sizes"], config["num_filters"]).to(
    device)
model_2.load_state_dict(torch.load(os.path.join(model_dir, "model_2_stage2.pth"), map_location=device))
model_2.eval()


# ================= 3. 数据处理函数 =================
def encode_texts(texts):
    """批量将文本转换为 ID 张量"""
    max_len = config["MAX_LEN"]
    encoded_list = []
    for text in texts:
        encoded = [vocab.get(char, vocab.get("<UNK>", 1)) for char in str(text)]
        if len(encoded) < max_len:
            encoded += [vocab.get("<PAD>", 0)] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        encoded_list.append(encoded)
    return torch.tensor(encoded_list, dtype=torch.long)


# ================= 4. 读取所有 CSV 的最后 10% =================
print("正在提取各 CSV 文件的最后 10% 数据...")
files = [
    ('Telecom_Fraud_Texts_5-main/label00-last.csv', 0),
    ('Telecom_Fraud_Texts_5-main/label01-last.csv', 1),
    ('Telecom_Fraud_Texts_5-main/label02-last.csv', 2),
    ('Telecom_Fraud_Texts_5-main/label03-last.csv', 3),
    ('Telecom_Fraud_Texts_5-main/label04-last.csv', 4)
]

all_texts = []
y_true = []

for f, label in files:
    if not os.path.exists(f):
        print(f"警告: 找不到文件 {f}，已跳过。")
        continue

    try:
        df = pd.read_csv(f, encoding='utf-8')
    except:
        df = pd.read_csv(f, encoding='gbk')

    # 取最后 10% 的数据
    tail_count = int(len(df) * 0.1)
    df_tail = df.tail(tail_count)

    all_texts.extend(df_tail['content'].tolist())
    y_true.extend([label] * tail_count)
    print(f" - {f}: 提取了 {tail_count} 条 (总数据量的 10%)")

# ================= 5. 批量推理预测 =================
print(f"\n开始对 {len(all_texts)} 条数据进行双阶段批量推理...")
tensor_inputs = encode_texts(all_texts)
dataset = TensorDataset(tensor_inputs)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

y_pred = []
THRESHOLD = config["threshold"]  # 使用之前训练时保存的 0.85 拦截阈值

with torch.no_grad():
    for batch in dataloader:
        x = batch[0].to(device)

        # 阶段一：二分类 (正常 vs 诈骗)
        out_1 = model_1(x)
        probs_1 = torch.softmax(out_1, dim=1)
        is_scam = (probs_1[:, 1] > THRESHOLD).cpu().numpy()

        # 初始化当前 Batch 的预测结果（默认为 0 正常）
        batch_preds = np.zeros(len(x), dtype=int)

        # 找出被判定为诈骗的索引
        scam_indices = np.where(is_scam)[0]

        if len(scam_indices) > 0:
            scam_x = x[scam_indices]
            # 阶段二：具体诈骗四分类
            out_2 = model_2(scam_x)
            preds_2 = torch.argmax(out_2, dim=1).cpu().numpy()

            # 将阶段二的分类结果（0~3）映射回原标签（1~4）
            for idx, scam_type in zip(scam_indices, preds_2):
                batch_preds[idx] = scam_type + 1

        y_pred.extend(batch_preds.tolist())

# ================= 6. 打印评估报告 =================
print("\n" + "=" * 50)
print("🚀 最后 10% 数据测试完毕！验证报告如下：")
print("=" * 50)

target_names = config["target_names"]
print(classification_report(y_true, y_pred, target_names=target_names))

print("混淆矩阵 (行:真实标签, 列:预测标签):")
cm = confusion_matrix(y_true, y_pred)
# 简单格式化打印混淆矩阵以便阅读
cm_df = pd.DataFrame(cm, index=[f"真:{n[:2]}" for n in target_names], columns=[f"预:{n[:2]}" for n in target_names])
print(cm_df)