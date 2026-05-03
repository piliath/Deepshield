import os
import pandas as pd
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset

try:
    from mindnlp.models import BertForSequenceClassification, BertConfig
    from mindnlp.transforms import BertTokenizer
except ImportError:
    print("警告：请安装 mindnlp 以支持 BERT 模型 (pip install mindnlp)")
    exit()

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU") # 或根据实际情况修改为 "Ascend" / "GPU"

def load_data(data_dir="Telecom_Fraud_Texts_5-main"):
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
        df['text'] = df['content'].astype(str)
        # 标签处理：阶段一 (二分类)，阶段二 (四分类)
        df['stage1_label'] = 0 if idx == 0 else 1
        df['stage2_label'] = -1 if idx == 0 else idx - 1
        
        split_idx = int(len(df) * 0.9)
        train_dfs.append(df.iloc[:split_idx])
        test_dfs.append(df.iloc[split_idx:])

    if not train_dfs:
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)

class BERTDatasetGenerator:
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # 使用 MindNLP 的 Tokenizer
        tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len
        )
        return (
            np.array(tokens['input_ids'], dtype=np.int32),
            np.array(tokens['attention_mask'], dtype=np.int32),
            np.array(self.labels[idx], dtype=np.int32)
        )

    def __len__(self):
        return len(self.texts)

def main():
    print("🚀 初始化 BERT-base (MindSpore / MindNLP 版本)...")
    train_df, test_df = load_data()
    if len(train_df) == 0:
        print("未找到数据，请检查 Telecom_Fraud_Texts_5-main 目录！")
        return
        
    model_name = 'hfl/chinese-macbert-base'
    print(f"正在加载分词器: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # ------------------------------------------
    # 阶段一：正常 vs 诈骗 二分类
    # ------------------------------------------
    print("\n========== 配置阶段一 (二分类) 网络 ==========")
    train_texts_s1 = train_df['text'].tolist()
    train_labels_s1 = train_df['stage1_label'].tolist()
    
    # 构建 Dataset
    dataset_generator_s1 = BERTDatasetGenerator(train_texts_s1, train_labels_s1, tokenizer)
    ms_dataset_s1 = GeneratorDataset(
        source=dataset_generator_s1, 
        column_names=["input_ids", "attention_mask", "labels"],
        shuffle=True
    ).batch(16)
    
    # 实例化预训练模型 (设置 config 参数强制关闭 weights 拉取中的讨论区检查等潜在问题)
    config_s1 = BertConfig.from_pretrained(model_name, num_labels=2)
    model_s1 = BertForSequenceClassification(config_s1)
    
    # 【代价敏感学习】: 给0(正常)设置惩罚权重为5.0，1(诈骗)为1.0
    loss_fn_s1 = nn.CrossEntropyLoss(weight=ms.Tensor([5.0, 1.0], ms.float32))
    optimizer_s1 = nn.AdamWeightDecay(model_s1.trainable_params(), learning_rate=2e-5)
    
    net_with_loss_s1 = nn.WithLossCell(model_s1, loss_fn_s1)
    train_net_s1 = nn.TrainOneStepCell(net_with_loss_s1, optimizer_s1)
    
    print("阶段一模型网络结构配置成功。")

    # ------------------------------------------
    # 阶段二：具体诈骗四分类
    # ------------------------------------------
    print("\n========== 配置阶段二 (四分类) 网络 ==========")
    scam_train_df = train_df[train_df['stage1_label'] == 1]
    train_texts_s2 = scam_train_df['text'].tolist()
    train_labels_s2 = scam_train_df['stage2_label'].tolist()
    
    dataset_generator_s2 = BERTDatasetGenerator(train_texts_s2, train_labels_s2, tokenizer)
    ms_dataset_s2 = GeneratorDataset(
        source=dataset_generator_s2, 
        column_names=["input_ids", "attention_mask", "labels"],
        shuffle=True
    ).batch(16)
    
    config_s2 = BertConfig.from_pretrained(model_name, num_labels=4)
    model_s2 = BertForSequenceClassification(config_s2)
    
    loss_fn_s2 = nn.CrossEntropyLoss()
    optimizer_s2 = nn.AdamWeightDecay(model_s2.trainable_params(), learning_rate=2e-5)
    
    net_with_loss_s2 = nn.WithLossCell(model_s2, loss_fn_s2)
    train_net_s2 = nn.TrainOneStepCell(net_with_loss_s2, optimizer_s2)

    print("阶段二模型网络结构配置成功。")
    print("\n✅ 所有的基于 MindSpore 的 BERT 训练流水线已构建完毕！")
    print("可调用以下代码开启前向训练：")
    print("for input_ids, mask, label in ms_dataset_s1.create_tuple_iterator():")
    print("    loss = train_net_s1(input_ids, label)")
    print("    print('Batch Loss:', loss)")

if __name__ == "__main__":
    main()
