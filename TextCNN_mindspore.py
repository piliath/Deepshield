import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import numpy as np
import pandas as pd
import os
import json

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU") # 建议根据实际硬件修改为"GPU"或"Ascend"

# 1. 加载数据并处理编码
def load_data(data_dir="Telecom_Fraud_Texts_5-main"):
    files = [
        ('label00-last.csv', 0),
        ('label01-last.csv', 1),
        ('label02-last.csv', 2),
        ('label03-last.csv', 3),
        ('label04-last.csv', 4)
    ]
    dfs = []
    for f, label in files:
        file_path = os.path.join(data_dir, f)
        if not os.path.exists(file_path):
            continue
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='gbk')
        
        train_count = int(len(df) * 0.9)
        df = df.head(train_count)
        df['text'] = df['content'].astype(str)
        df['label_id'] = label
        dfs.append(df[['text', 'label_id']])
    
    if not dfs:
        return pd.DataFrame(columns=['text', 'label_id'])
    return pd.concat(dfs, ignore_index=True)

class TextDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return np.array(self.texts[index], dtype=np.int32), np.array(self.labels[index], dtype=np.int32)

    def __len__(self):
        return len(self.labels)

# 2. 定义TextCNN模型
class TextCNN(nn.Cell):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=[2, 3, 4], num_filters=64):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Conv1d in MindSpore expects (batch_size, in_channels, seq_length)
        self.convs = nn.CellList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs, pad_mode='pad')
            for fs in filter_sizes
        ])
        self.fc = nn.Dense(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(p=0.5) # keep_prob = 1 - p
        self.relu = nn.ReLU()
        self.reduce_max = ms.ops.ReduceMax(keep_dims=False)

    def construct(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.transpose((0, 2, 1))  # [batch, embed_dim, seq_len]

        pooled_outputs = []
        for conv in self.convs:
            c = self.relu(conv(x))
            p = self.reduce_max(c, 2) # Max pool over sequence length
            pooled_outputs.append(p)

        h_pool = ms.ops.concat(pooled_outputs, axis=1)
        h_pool = self.dropout(h_pool)
        out = self.fc(h_pool)
        return out

if __name__ == "__main__":
    print("开始加载并训练TextCNN (MindSpore版本)...")
    data = load_data()
    if data.empty:
        print("未找到数据文件，演示模式启动。")
        vocab_size = 1000
    else:
        # 建立词表等操作...
        vocab_size = 5000
    
    embed_dim = 64
    model_1 = TextCNN(vocab_size, embed_dim, num_classes=2)
    loss_fn = nn.CrossEntropyLoss(weight=ms.Tensor([5.0, 1.0], ms.float32))
    optimizer = nn.Adam(model_1.trainable_params(), learning_rate=0.005)
    
    # 定义训练网络
    net_with_loss = nn.WithLossCell(model_1, loss_fn)
    train_net = nn.TrainOneStepCell(net_with_loss, optimizer)
    
    print("模型初始化完成，准备进行训练...")
    # 训练循环 (此处为伪代码骨架，具体需要配合DataLoader)
    # for data, label in dataset.create_tuple_iterator():
    #     loss = train_net(data, label)
    #     print("Loss:", loss.asnumpy())
