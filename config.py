import os
import json

# 文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAILBOXES_FILE = os.path.join(DATA_DIR, "mailboxes.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
REPORTS_FILE = os.path.join(DATA_DIR, "reports.json")
STATS_FILE = os.path.join(DATA_DIR, "stats.json")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 默认设置
DEFAULT_SETTINGS = {
    "dashscope_api_key": "",
    "fetch_interval": 300,
    "use_qwen3_model": False
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_SETTINGS

def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

def load_json(file_path, default=[]):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 模型配置
LIGHTGBM_DIR = "LightGBM+TF-IDF"
TEXTCNN_DIR = "TextCNN"
BERT_DIR = "BERT-base"
QWEN3_BASE = os.path.join(BASE_DIR, "Qwen3-0___6B") # 基础模型路径
QWEN3_DIR = os.path.join("qwen3_lora_fraud", "checkpoint-2650") # 微调后的 LoRA 路径
FRAUD_THRESHOLD = 0.85

# 邮箱服务默认配置
IMAP_HOST = "imap.163.com"
IMAP_PORT = 993
SMTP_HOST = "smtp.163.com"
SMTP_PORT = 465
