import os
import json
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np

# 设置 HuggingFace 国内镜像，并关闭无用的后台讨论区检查日志
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 屏蔽 auto_conversion 线程的 403 报错
os.environ["HF_HUB_OFFLINE"] = "1" # 强制使用本地文件，不再尝试连接官网

# 推理时所需的重型库放在内部加载，避免环境缺失导致整个引擎无法启动
# From core import ...

# ================= 1. TextCNN 模型结构定义 =================
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

# ================= 2. 核心检测引擎 =================
class FraudDetectionEngine:
    def __init__(self, config_obj):
        self.cfg = config_obj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型缓存
        self.models = {
            "lgbm": None,
            "textcnn": None,
            "bert": None,
            "qwen3": None
        }
        
    # --- LightGBM 加载与推理 ---
    def _load_lgbm(self):
        if self.models["lgbm"] is None:
            path = self.cfg.LIGHTGBM_DIR
            self.models["lgbm"] = {
                "tfidf": joblib.load(os.path.join(path, 'tfidf_vectorizer.pkl')),
                "s1": joblib.load(os.path.join(path, 'clf_stage1.pkl')),
                "s2": joblib.load(os.path.join(path, 'clf_stage2.pkl')),
                "id_to_label": joblib.load(os.path.join(path, 'id_to_label.pkl'))
            }
            print("[Engine] LightGBM 模型加载成功")
            
    def predict_lgbm(self, text):
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        
        self._load_lgbm()
        m = self.models["lgbm"]
        X = m["tfidf"].transform([text])
        
        # 阶段一：二分类
        # 确保带上特征名，或者静默警告
        prob_scam = m["s1"].predict_proba(X)[0][1]
        
        # 阶段二：多分类
        pred_id = m["s2"].predict(X)[0]
        type_name = m["id_to_label"].get(pred_id, "未知诈骗")
        
        return prob_scam, type_name

    # --- TextCNN 加载与推理 ---
    def _load_textcnn(self):
        if self.models["textcnn"] is None:
            path = self.cfg.TEXTCNN_DIR
            with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
                t_cfg = json.load(f)
            with open(os.path.join(path, "vocab.json"), "r", encoding="utf-8") as f:
                vocab = json.load(f)
                
            s1 = TextCNN(t_cfg["vocab_size"], t_cfg["embed_dim"], 2, t_cfg["filter_sizes"], t_cfg["num_filters"]).to(self.device)
            s1.load_state_dict(torch.load(os.path.join(path, "model_1_stage1.pth"), map_location=self.device))
            s1.eval()
            
            s2 = TextCNN(t_cfg["vocab_size"], t_cfg["embed_dim"], 4, t_cfg["filter_sizes"], t_cfg["num_filters"]).to(self.device)
            s2.load_state_dict(torch.load(os.path.join(path, "model_2_stage2.pth"), map_location=self.device))
            s2.eval()
            
            self.models["textcnn"] = {"s1": s1, "s2": s2, "vocab": vocab, "cfg": t_cfg}
            print("[Engine] TextCNN 模型加载成功")

    def predict_textcnn(self, text):
        self._load_textcnn()
        m = self.models["textcnn"]
        
        # 文本转ID
        max_len = m["cfg"]["MAX_LEN"]
        vocab = m["vocab"]
        encoded = [vocab.get(char, vocab.get("<UNK>", 1)) for char in str(text)]
        if len(encoded) < max_len:
            encoded += [vocab.get("<PAD>", 0)] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        
        x = torch.tensor([encoded], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # 阶段一
            out1 = m["s1"](x)
            prob_scam = torch.softmax(out1, dim=1)[0][1].item()
            
            # 阶段二
            out2 = m["s2"](x)
            type_idx = torch.argmax(out2, dim=1).item()
            type_name = m["cfg"]["target_names"][type_idx + 1] # 映射回1-4
            
        return prob_scam, type_name

    # --- BERT 加载与推理 ---
    def _load_bert(self):
        # 局部导入 transformers，解决启动时的依赖报错
        import transformers
        # 强力屏蔽 transformers 中自动转换 safetensors 的后台线程产生的 403 报错
        if hasattr(transformers, "safetensors_conversion"):
            transformers.safetensors_conversion.auto_conversion = lambda *args, **kwargs: None

        from transformers import BertTokenizer, BertForSequenceClassification
        
        if self.models["bert"] is None:
            path = self.cfg.BERT_DIR
            base_model = "hfl/chinese-macbert-base"
            print(f"[Engine] BERT 正在从 {base_model} 加载骨架...")
            
            tokenizer = BertTokenizer.from_pretrained(path)
            
            # 加载并修正权重逻辑 (处理 DataParallel 保存时带 module. 的情况)
            def load_and_fix_weights(model, weight_path):
                state_dict = torch.load(weight_path, map_location=self.device)
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 关键修复：处理各种前缀不匹配
                    name = k.replace('module.', '').replace('bert.bert.', 'bert.')
                    new_state_dict[name] = v
                
                # 尝试加载
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                if missing:
                    print(f"[Engine] BERT 加载警告 - 以下权重缺失: {missing[:3]}...")
                if unexpected:
                    print(f"[Engine] BERT 加载警告 - 以下权重多余: {unexpected[:3]}...")
                
                model.eval()
                return model.to(self.device)

            s1 = BertForSequenceClassification.from_pretrained(base_model, num_labels=2)
            s1 = load_and_fix_weights(s1, os.path.join(path, 'stage1_weights.pth'))
            
            s2 = BertForSequenceClassification.from_pretrained(base_model, num_labels=4)
            s2 = load_and_fix_weights(s2, os.path.join(path, 'stage2_weights.pth'))
            
            self.models["bert"] = {"s1": s1, "s2": s2, "tokenizer": tokenizer}
            print("[Engine] BERT 两阶段权重注入成功")

    def predict_bert(self, text):
        self._load_bert()
        m = self.models["bert"]
        inputs = m["tokenizer"](text, return_tensors="pt", max_length=128, truncation=True, padding='max_length').to(self.device)
        
        with torch.no_grad():
            # 阶段一
            out1 = m["s1"](**inputs)
            prob_scam = torch.softmax(out1.logits, dim=-1)[0][1].item()
            
            # 阶段二
            out2 = m["s2"](**inputs)
            type_idx = torch.argmax(out2.logits, dim=-1).item()
            # 映射标签
            labels = ["公检法诈骗", "贷款诈骗", "客服诈骗", "熟人诈骗"]
            type_name = labels[type_idx]
            
        return prob_scam, type_name

    # --- Qwen3-0.6B 加载与推理 ---
    def _load_qwen3(self):
        # 局部导入 transformers 和 peft，解决启动时的依赖报错
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        if self.models["qwen3"] is None:
            # 强化本地路径识别，解决 Repo id 报错
            raw_base_path = self.cfg.QWEN3_BASE 
            base_path = os.path.abspath(raw_base_path) # 转换为绝对路径，避免被误认为 Repo ID
            
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"找不到本地大模型路径，请检查 config.py 配置是否正确或模型是否缺失: {base_path}")
                
            lora_path = os.path.join(self.cfg.BASE_DIR, self.cfg.QWEN3_DIR)
            
            print(f"[Engine] Qwen3 正在从本地路径加载: {base_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, local_files_only=True)
            model = PeftModel.from_pretrained(model, lora_path)
            model.eval()
            
            self.models["qwen3"] = {"model": model, "tokenizer": tokenizer}
            print("[Engine] Qwen3-0.6B LoRA 模型加载成功")

    def predict_qwen3(self, text):
        self._load_qwen3()
        m = self.models["qwen3"]
        tokenizer = m["tokenizer"]
        
        # 使用 apply_chat_template 准备对话格式 (见微调代码)
        messages = [{"role": "user", "content": text}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = m["model"].generate(**inputs, max_new_tokens=20, do_sample=False)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()
            
        labels = ["公检法诈骗", "贷款诈骗", "客服诈骗", "熟人诈骗"]
        is_fraud = any(l in pred for l in labels)
        
        # 如果模型回的是标签索引或者更简单的字符串
        if not is_fraud:
            if "诈骗" in pred: 
                is_fraud = True
                pred = "疑似诈骗"
        
        return (1.0 if is_fraud else 0.0), (pred if is_fraud else "正常")

    # --- 全模型集成预警 ---
    def detect(self, text, use_qwen3=False):
        results = {
            "is_fraud": False,
            "trigger_models": [],
            "details": {}
        }
        
        # 1. 运行三个基础模型
        m_list = [
            ("LightGBM", self.predict_lgbm),
            ("TextCNN", self.predict_textcnn),
            ("BERT", self.predict_bert)
        ]
        
        if use_qwen3:
            m_list.append(("Qwen3", self.predict_qwen3))
            
        for name, func in m_list:
            try:
                prob, t_name = func(text)
                results["details"][name] = {"prob": prob, "type": t_name}
                if prob >= self.cfg.FRAUD_THRESHOLD:
                    results["is_fraud"] = True
                    results["trigger_models"].append(name)
            except Exception as e:
                print(f"[Engine] {name} 推理失败: {str(e)}")
                
        return results
