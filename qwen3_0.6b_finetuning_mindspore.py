import mindspore as ms
from mindspore import nn

# 注意：运行此代码需要安装 mindformers
# pip install mindformers
try:
    from mindformers import AutoModelForCausalLM, AutoTokenizer
    from mindformers import TrainingArguments, Trainer
    from mindformers.pet import LoraConfig, get_pet_model
except ImportError:
    print("警告：请安装 mindformers 以支持大模型微调机制。")

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend") # 大模型建议使用Ascend或GPU

def main():
    print("🚀 开始配置 Qwen-0.6B LoRA 微调 (MindSpore/MindFormers 版本)...")
    
    MODEL_PATH = "qwen/qwen-0_6b"  # 在MindFormers中对应的模型名称或本地路径
    
    try:
        # 1. 加载 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 2. 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        
        # 3. LoRA 配置
        # MindFormers的PET库支持LoRA
        lora_config = LoraConfig(
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules='.*.(wq|wk|wv|wo)' # 对应 q_proj, k_proj, v_proj, o_proj
        )
        
        # 获取注入LoRA的模型
        model = get_pet_model(model, lora_config)
        print("✅ LoRA 配置成功，可训练参数已调整。")
        
        # 4. 配置 TrainingArguments
        training_args = TrainingArguments(
            output_dir="./qwen3_lora_fraud_ms",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=2,
            logging_steps=20,
            save_steps=200,
            evaluation_strategy="steps",
            eval_steps=200,
            save_total_limit=3
        )
        
        # 5. 初始化Trainer并训练
        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset
        # )
        # trainer.train()
        print("✅ Trainer 配置就绪！(需要传入 Dataset 即可开始 train())")
        
    except Exception as e:
        print("初始化大模型时遇到问题，通常是缺少 mindformers 或模型权重未下载。")
        print(f"详情: {e}")

if __name__ == "__main__":
    main()
