import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os
import json

warnings.filterwarnings('ignore')

# 1. 加载数据并处理编码 (只取每个文件的前 90%)
files = [
    ('Telecom_Fraud_Texts_5-main/label00-last.csv', 0),
    ('Telecom_Fraud_Texts_5-main/label01-last.csv', 1),
    ('Telecom_Fraud_Texts_5-main/label02-last.csv', 2),
    ('Telecom_Fraud_Texts_5-main/label03-last.csv', 3),
    ('Telecom_Fraud_Texts_5-main/label04-last.csv', 4)
]
dfs = []
for f, label in files:
    try:
        df = pd.read_csv(f, encoding='utf-8')
    except:
        df = pd.read_csv(f, encoding='gbk')

    # ================= 新增修改 =================
    # 计算前 90% 的数据量，并使用 .head() 截取
    train_count = int(len(df) * 0.9)
    df = df.head(train_count)
    print(f"文件 {f}: 总数读取前 90%，共 {train_count} 条数据进入训练池")
    # ============================================

    df['text'] = df['content'].astype(str)
    df['label_id'] = label  # 0是正常，1-4是不同诈骗
    dfs.append(df[['text', 'label_id']])

data = pd.concat(dfs, ignore_index=True)

# 2. 构建字符级词表 (Char-level Vocabulary，中文场景下往往比词级更鲁棒)
vocab = {"<PAD>": 0, "<UNK>": 1}
for text in data['text']:
    for char in text:
        if char not in vocab:
            vocab[char] = len(vocab)

MAX_LEN = 120  # 短信通常不长，截断长度设定为120


def encode_text(text):
    encoded = [vocab.get(char, vocab["<UNK>"]) for char in text]
    if len(encoded) < MAX_LEN:
        encoded += [vocab["<PAD>"]] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]
    return encoded


data['encoded'] = data['text'].apply(encode_text)

# 阶段一标签: 0=正常，1=诈骗 (把1~4全归为1)
data['stage1_label'] = data['label_id'].apply(lambda x: 0 if x == 0 else 1)

# 拆分训练集和测试集 (这里是在前90%的数据中再切分80%训练、20%验证，保持各类分布比例 stratify)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label_id'])


# 为阶段一和阶段二准备 DataLoader
class TextDataset(Dataset):
    def __init__(self, df, label_col):
        self.texts = torch.tensor(df['encoded'].tolist(), dtype=torch.long)
        self.labels = torch.tensor(df[label_col].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# 阶段一 DataLoader (全量数据)
train_stage1 = TextDataset(train_data, 'stage1_label')
test_stage1 = TextDataset(test_data, 'stage1_label')
loader_train_1 = DataLoader(train_stage1, batch_size=64, shuffle=True)
loader_test_1 = DataLoader(test_stage1, batch_size=64, shuffle=False)

# 阶段二 DataLoader (仅过滤出的诈骗数据，标签从1-4映射为0-3)
train_scam = train_data[train_data['stage1_label'] == 1].copy()
train_scam['stage2_label'] = train_scam['label_id'] - 1

train_stage2 = TextDataset(train_scam, 'stage2_label')
loader_train_2 = DataLoader(train_stage2, batch_size=64, shuffle=True)


# 3. 定义 TextCNN 模型结构
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=[2, 3, 4], num_filters=64):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 多尺寸一维卷积核提取局部特征（对应N-gram）
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # TextCNN要求维度顺序 [batch, channels(embed_dim), seq_len]

        pooled_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            p = torch.max_pool1d(c, c.size(2)).squeeze(2)  # 最大池化捕捉最强特征
            pooled_outputs.append(p)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool = self.dropout(h_pool)
        out = self.fc(h_pool)
        return out


# 4. 训练准备
vocab_size = len(vocab)
embed_dim = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 阶段一：二分类训练 =================
print("正在训练阶段一模型 (正常 vs 诈骗)...")
model_1 = TextCNN(vocab_size, embed_dim, num_classes=2).to(device)

# 【核心策略 1：代价敏感学习】
# 设定交叉熵损失的类别权重。模型要是错将正常(类别0)判为诈骗(类别1)，损失惩罚是普通错误的 5 倍！
class_weights = torch.tensor([5.0, 1.0]).to(device)
criterion_1 = nn.CrossEntropyLoss(weight=class_weights)
optimizer_1 = optim.Adam(model_1.parameters(), lr=0.005)

for epoch in range(5):
    model_1.train()
    for texts, labels in loader_train_1:
        texts, labels = texts.to(device), labels.to(device)
        optimizer_1.zero_grad()
        loss = criterion_1(model_1(texts), labels)
        loss.backward()
        optimizer_1.step()

# ================= 阶段二：具体诈骗类型训练 =================
print("正在训练阶段二模型 (具体诈骗类型识别)...")
model_2 = TextCNN(vocab_size, embed_dim, num_classes=4).to(device)
criterion_2 = nn.CrossEntropyLoss()
optimizer_2 = optim.Adam(model_2.parameters(), lr=0.005)

for epoch in range(10):
    model_2.train()
    for texts, labels in loader_train_2:
        texts, labels = texts.to(device), labels.to(device)
        optimizer_2.zero_grad()
        loss = criterion_2(model_2(texts), labels)
        loss.backward()
        optimizer_2.step()

# ================= 流水线评估与预测 =================
model_1.eval()
model_2.eval()

y_true_all = test_data['label_id'].tolist()
y_pred_all = []

# 【核心策略 2：调整决策阈值】
# 模型正常认定 > 0.5 就是诈骗，我们将其提高至 0.85，进一步严防误杀。
THRESHOLD = 0.85

test_texts_tensor = torch.tensor(test_data['encoded'].tolist(), dtype=torch.long).to(device)

with torch.no_grad():
    # 步骤 A: 走阶段一模型
    outputs_1 = model_1(test_texts_tensor)
    probs_1 = torch.softmax(outputs_1, dim=1)

    # 只要诈骗概率 > THRESHOLD，才判定为疑似诈骗
    is_scam = (probs_1[:, 1] > THRESHOLD).cpu().numpy()

    # 步骤 B: 将疑似诈骗的数据送入阶段二分类器
    scam_indices = np.where(is_scam)[0]
    if len(scam_indices) > 0:
        scam_texts = test_texts_tensor[scam_indices]
        outputs_2 = model_2(scam_texts)
        preds_2 = torch.argmax(outputs_2, dim=1).cpu().numpy()

    # 步骤 C: 组装最终结果
    for i in range(len(test_texts_tensor)):
        if not is_scam[i]:
            y_pred_all.append(0)  # 判定为正常
        else:
            idx_in_scam = np.where(scam_indices == i)[0][0]
            final_label = preds_2[idx_in_scam] + 1  # 映射回1-4的标签
            y_pred_all.append(final_label)

# 打印报告
target_names = ['正常 (0)', '公检法诈骗 (1)', '贷款诈骗 (2)', '客服诈骗 (3)', '熟人诈骗 (4)']
print("\n========== 最终分类报告 (Two-Stage Pipeline, 前90%数据) ==========")
print(classification_report(y_true_all, y_pred_all, target_names=target_names))
print("混淆矩阵:\n", confusion_matrix(y_true_all, y_pred_all))

# ================= 封存模型与配置 =================
save_dir = "TextCNN"

# 1. 创建目标文件夹
os.makedirs(save_dir, exist_ok=True)

# 2. 保存 PyTorch 模型权重
torch.save(model_1.state_dict(), os.path.join(save_dir, "model_1_stage1.pth"))
torch.save(model_2.state_dict(), os.path.join(save_dir, "model_2_stage2.pth"))

# 3. 保存字符级词表 (Vocab)
with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

# 4. 保存模型超参数 (Config)
config = {
    "MAX_LEN": MAX_LEN,
    "vocab_size": vocab_size,
    "embed_dim": embed_dim,
    "filter_sizes": [2, 3, 4],
    "num_filters": 64,
    "threshold": THRESHOLD,
    "target_names": ['正常 (0)', '公检法诈骗 (1)', '贷款诈骗 (2)', '客服诈骗 (3)', '熟人诈骗 (4)']
}
with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print(f"\n✅ 恭喜！基于前 90% 数据训练的模型及配置已成功打包并保存至文件夹: ./{save_dir}/")