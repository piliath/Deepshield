import os

# 设置 HuggingFace 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 symlink 警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 【关键】禁用 safetensors 自动转换，避免 403 错误
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. 数据读取与切分 (90% 训练, 10% 测试)
# ==========================================
def load_and_split_data(file_paths):
    train_dfs, test_dfs = [], []

    for idx, f in enumerate(file_paths):
        # 兼容 utf-8 和 gb18030(包含gbk)
        try:
            df = pd.read_csv(f, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding='gb18030')
        except FileNotFoundError:
            print(f"文件不存在: {f}")
            continue

        n = len(df)
        split_idx = int(n * 0.9)

        if 'label00' in f:
            df['stage1_label'] = 0
            df['stage2_label'] = -1  # 正常短信不参与第二阶段
        else:
            df['stage1_label'] = 1
            df['stage2_label'] = idx - 1  # label01 -> 0, label02 -> 1...

        train_dfs.append(df.iloc[:split_idx])
        test_dfs.append(df.iloc[split_idx:])

    if not train_dfs:
        raise ValueError("没有找到任何数据文件")

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    return train_df, test_df


# ==========================================
# 2. Dataset 与 代价敏感学习 Trainer
# ==========================================
class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 修复：直接使用 tokenizer，而不是 encode_plus
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class CostSensitiveTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ==========================================
# 3. 核心训练与统一模型保存
# ==========================================
def main():
    # 检查文件是否存在
    file_paths = [
        'Telecom_Fraud_Texts_5-main/label00-last.csv',
        'Telecom_Fraud_Texts_5-main/label01-last.csv',
        'Telecom_Fraud_Texts_5-main/label02-last.csv',
        'Telecom_Fraud_Texts_5-main/label03-last.csv',
        'Telecom_Fraud_Texts_5-main/label04-last.csv'
    ]

    # 检查文件是否存在
    missing_files = [f for f in file_paths if not os.path.exists(f)]
    if missing_files:
        print(f"警告: 以下文件不存在: {missing_files}")
        print("请确保数据文件在正确的路径下")

    print("正在加载数据...")
    train_df, test_df = load_and_split_data(file_paths)
    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
    print(
        f"训练集中正常样本: {(train_df['stage1_label'] == 0).sum()}, 诈骗样本: {(train_df['stage1_label'] == 1).sum()}")

    model_name = 'hfl/chinese-macbert-base'

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载 tokenizer，添加一些参数避免警告
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 创建一个统一的输出文件夹
    SAVE_DIR = './BERT-base'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 提前把通用的分词器词表存进该文件夹
    tokenizer.save_pretrained(SAVE_DIR)

    # ------------- 阶段一：正常 vs 诈骗 -------------
    print("\n========== 开始训练阶段一 (二分类) ==========")
    train_texts_s1 = train_df['content'].tolist()
    train_labels_s1 = train_df['stage1_label'].tolist()
    ds_train_s1 = SMSDataset(train_texts_s1, train_labels_s1, tokenizer)

    model_s1 = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args_s1 = TrainingArguments(
        output_dir='./tmp_s1',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_steps=50,
        save_strategy='no',
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # 添加这个参数避免 pin_memory 警告
    )

    class_weights_s1 = torch.tensor([5.0, 1.0], dtype=torch.float)

    trainer_s1 = CostSensitiveTrainer(
        class_weights=class_weights_s1,
        model=model_s1,
        args=args_s1,
        train_dataset=ds_train_s1
    )

    print("开始训练阶段一...")
    trainer_s1.train()

    # 保存阶段一的权重
    torch.save(model_s1.state_dict(), os.path.join(SAVE_DIR, 'stage1_weights.pth'))
    print(f"阶段一权重已独立保存至 {SAVE_DIR}/stage1_weights.pth")

    # ------------- 阶段二：诈骗具体4分类 -------------
    print("\n========== 开始训练阶段二 (四分类) ==========")
    scam_train_df = train_df[train_df['stage1_label'] == 1]
    print(f"诈骗短信训练样本数: {len(scam_train_df)}")

    if len(scam_train_df) == 0:
        print("警告: 没有找到诈骗短信样本，跳过阶段二训练")
        return

    # 检查各类别样本数量
    print("诈骗类别分布:")
    for i in range(4):
        count = (scam_train_df['stage2_label'] == i).sum()
        print(f"  类别 {i}: {count} 样本")

    train_texts_s2 = scam_train_df['content'].tolist()
    train_labels_s2 = scam_train_df['stage2_label'].tolist()
    ds_train_s2 = SMSDataset(train_texts_s2, train_labels_s2, tokenizer)

    model_s2 = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)

    args_s2 = TrainingArguments(
        output_dir='./tmp_s2',
        num_train_epochs=4,
        per_device_train_batch_size=16,
        logging_steps=50,
        save_strategy='no',
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # 添加这个参数避免 pin_memory 警告
    )

    trainer_s2 = CostSensitiveTrainer(
        class_weights=None,
        model=model_s2,
        args=args_s2,
        train_dataset=ds_train_s2
    )

    print("开始训练阶段二...")
    trainer_s2.train()

    # 保存阶段二的权重
    torch.save(model_s2.state_dict(), os.path.join(SAVE_DIR, 'stage2_weights.pth'))
    print(f"阶段二权重已独立保存至 {SAVE_DIR}/stage2_weights.pth")

    # ------------- 模型推理与评估 -------------
    print("\n========== 开始测试集评估 ==========")
    model_s1.eval()
    model_s2.eval()
    model_s1.to(device)
    model_s2.to(device)

    final_preds = []
    true_labels = []

    THRESHOLD_SCAM = 0.85

    print("正在进行预测...")
    for idx, row in test_df.iterrows():
        text = str(row['content'])
        true_stage1 = row['stage1_label']
        true_stage2 = row['stage2_label']

        actual_class = 0 if true_stage1 == 0 else true_stage2 + 1
        true_labels.append(actual_class)

        # 修复：直接使用 tokenizer
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding='max_length'
        ).to(device)

        with torch.no_grad():
            outputs_s1 = model_s1(**inputs)
            probs_s1 = torch.nn.functional.softmax(outputs_s1.logits, dim=-1)
            prob_scam = probs_s1[0][1].item()

            if prob_scam >= THRESHOLD_SCAM:
                outputs_s2 = model_s2(**inputs)
                pred_s2 = torch.argmax(outputs_s2.logits, dim=-1).item()
                final_preds.append(pred_s2 + 1)
            else:
                final_preds.append(0)

        # 每100条打印一次进度
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(test_df)} 条测试样本")

    print("\n========== 两阶段串联分类报告 ==========")
    target_names = ['正常(0)', '诈骗类型1(1)', '诈骗类型2(2)', '诈骗类型3(3)', '诈骗类型4(4)']
    print(classification_report(true_labels, final_preds, target_names=target_names, digits=4))

    # 计算准确率
    accuracy = (np.array(true_labels) == np.array(final_preds)).mean()
    print(f"\n总体准确率: {accuracy:.4f}")


if __name__ == "__main__":
    main()