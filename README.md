# 🛡️ 多模态全自动反诈预警系统 (Multi-Modal Anti-Fraud System)

本项目是一个基于 **Flask** 架构，集成 **多模态深度学习模型** 与 **大语言模型 (LLM)** 的自动化邮件/短信诈骗检测系统。它旨在为用户提供实时的、可解释的高精准反诈预警，并支持通过侧边栏管理受监控的 163 邮箱。

---

## 🚀 核心特性

- **多模态联合判定**：采用集成学习思想，整合四个维度模型进行预测：
  - **LightGBM + TF-IDF**：高效的传统机器学习统计模型。
  - **TextCNN**：捕捉局部特征的轻量级卷积神经网络。
  - **BERT (MacBERT-base)**：预训练语义理解专家，精准识别话术逻辑。
  - **Qwen3-0.6B (LoRA v2)**：经过微调的轻量级大模型，擅长场景定性。
- **LLM 深度报告**：接入 **Qwen-Plus** 接口，对拦截邮件自动生成人性化的“反诈定性报告”，包括骗局识别、原始摘要及防范建议。
- **全自动监控流水线**：后台线程每 5 分钟自动扫描一次配置的 163 邮箱，完成“抓取 -> 检测 -> 存档 -> 预警”闭环。
- **现代大厂级 UI**：基于鸿蒙字体 (HarmonyOS Sans SC) 设计，全站 SVG 图标化，支持平滑淡入动效及卡片悬浮反馈。
- **短信手动固存**：支持手动录入诈骗短信，自动生成训练语料库 (`data/sms_records.csv`)。

---

## 🏗️ 目录结构

```text
anticheat/
├── app.py                  # 系统入口 (Flask & Scheduler 启动)
├── config.py               # 全局配置 (模型路径、API Key、默认设置)
├── services/               # 核心业务层
│   ├── model_engine.py     # 多模态推理引擎 (权重加载与预测)
│   ├── email_fetcher.py    # 163 邮箱 IMAP 抓取 (带网易 ID 命令兼容)
│   ├── llm_reporter.py     # Qwen-Plus 报告生成模块
│   └── scheduler.py        # 后台自动化流水线调度
├── routes/                 # 蓝图路由 (统计API、邮箱管理、报告查询)
├── models/                 # 模型文件夹
│   ├── BERT-base/          # BERT 微调权重
│   ├── TextCNN/            # TextCNN 配置与权重
│   ├── LightGBM+TF-IDF/    # LGBM 序列化模型
│   └── Qwen3-0___6B/       # Qwen3 本地基础及 LoRA 权重
├── data/                   # 本地数据库 (JSON/CSV)
├── static/                 # 静态资源 (CSS/JS/Icons)
└── templates/              # 页面模板 (Dashboard/Reports/Settings)
```

---

## 🛠️ 安装与部署

### 1. 环境准备
建议在 Python 3.12+ 环境下运行。
```powershell
pip install -r requirements.txt
```

### 2. 模型权重放置
请确保 `BERT-base`, `TextCNN`, `LightGBM+TF-IDF` 以及 `Qwen3-0___6B` 文件夹已放置在根目录下或根据 `config.py` 中的路径进行调整。

### 3. 配置 API Key
1. 运行系统后进入“系统设置”。
2. 输入您的 **DashScope (Qwen) API Key**。
3. 可选：开启 Qwen3 本地联合判定（需要约 2GB 显存）。

---

## 📖 使用指南

### 1. 启动系统
```powershell
py app.py
```
访问 `http://127.0.0.1:5000` 即可进入控制面板。

### 2. 邮箱托管
1. 点击侧边栏 **“邮箱管理”**。
2. 输入您的 163 邮箱地址及其 **IMAP 授权码**。
3. 系统将立即启动首次同步，并开始每隔 5 分钟轮询。

### 3. 查看报告
一旦检出诈骗邮件，**“报告中心”** 会出现新卡片。点击可查看：
- **AI 定性分析**：大模型对该邮件骗局的详细解读。
- **模型分值**：观察 BERT、LGBM 等不同模型的置信度分布。
- **防范建议**：官方反诈话术提醒。

---

## ⚠️ 常见问题修复 (FAQ)

- **Q: BERT 加载显示 MISSING 权重？**
  - A: 这是 transformers 的警告日志。系统会在 `model_engine.py` 中通过 `load_and_fix_weights` 强制进行第二次手动覆盖加载，只要最终提示“权重注入成功”即可忽略警告。
- **Q: 403 Forbidden 报错？**
  - A: 系统已包含 Monkey-Patch 逻辑，强制屏蔽了 HuggingFace Hub 后台自动转换线程的冲突。请确保网络环境能正常使用或已开启隔离。
- **Q: Qwen3 提示 Repo id 格式错误？**
  - A: 这通常是因为配置的本地路径不存在。请检查 `config.py` 中 `QWEN3_BASE` 是否准确指向包含 `config.json` 的模型主目录。

---

## 🛡️ 免责声明
本系统仅用于诈骗辅助识别与学术交流，判定结果仅供参考，不作为法律依据。请时刻保持警惕，不轻信、不点击、不转账！
