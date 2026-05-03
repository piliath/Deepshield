import os
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ⭐ LoRA依赖
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ====================== 1. 配置 ======================
LOCAL_MODEL_PATH = r"D:\Qwen3-0___6B"
DATA_FOLDER = r"C:\Users\并没有\Desktop\Telecom_Fraud_Texts_5-main\Telecom_Fraud_Texts_5-main"

TEST_RATIO = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== 2. 数据加载 ======================
def load_all_csv(folder_path, test_ratio=0.1):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

    train_dfs, test_dfs = [], []

    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding="utf-8-sig", dtype=str, on_bad_lines="skip").fillna("")
        except:
            df = pd.read_csv(file, encoding="gbk", dtype=str, on_bad_lines="skip").fillna("")

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        split_idx = int(len(df) * (1 - test_ratio))
        train_dfs.append(df.iloc[:split_idx])
        test_dfs.append(df.iloc[split_idx:])

        print(f"✅ {file} -> 训练 {len(df.iloc[:split_idx])} | 测试 {len(df.iloc[split_idx:])}")

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
        "test_raw": Dataset.from_pandas(test_df)
    })

dataset_dict = load_all_csv(DATA_FOLDER, TEST_RATIO)
train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]
test_raw_dataset = dataset_dict["test_raw"]

# ====================== 3. tokenizer ======================
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    use_fast=False   # ⭐ 防炸
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ====================== 4. 模型 + LoRA ======================
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
    device_map="auto",
    trust_remote_code=True
)

# ⭐ LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)

# 打印可训练参数（确认LoRA生效）
model.print_trainable_parameters()

# ====================== 5. 数据处理 ======================
def format_data(example):
    user_content = example.get("content", "").strip() or "无内容"
    assistant_content = example.get("label", "").strip() or "无标签"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

train_dataset = train_dataset.map(format_data)
test_dataset = test_dataset.map(format_data)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

def clean_columns(dataset):
    return dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["input_ids", "attention_mask"]]
    )

tokenized_train = clean_columns(tokenized_train)
tokenized_test = clean_columns(tokenized_test)

# ====================== 6. 训练配置（LoRA优化版） ======================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./qwen3_lora_fraud",

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,
    num_train_epochs=2,

    logging_steps=20,
    save_steps=200,
    eval_steps=200,
    eval_strategy="steps",

    fp16=True,
    optim="adamw_torch",

    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)

# ====================== 7. 开始训练 ======================
print("\n🚀 开始 LoRA 微调...")
trainer.train()
print("✅ LoRA 微调完成！")

# ====================== 8. 准确率 ======================
def calculate_test_accuracy(test_raw_dataset, model, tokenizer):
    model.eval()
    correct, total = 0, 0

    for sample in test_raw_dataset:
        content = sample["content"].strip()
        truth = sample["label"].strip()

        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = pred.split("assistant")[-1].strip()

        total += 1
        if pred == truth or pred in truth or truth in pred:
            correct += 1

    return correct / total if total else 0


# ====================== 9. 评测 ======================
eval_results = trainer.evaluate()
accuracy = calculate_test_accuracy(test_raw_dataset, model, tokenizer)

print("\n📊 最终结果")
print(f"Loss: {eval_results['eval_loss']:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")