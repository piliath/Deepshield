import sys
import os
import time

# 确保能加载项目模块
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config
from services.model_engine import FraudDetectionEngine

def test_inference():
    print("=== 多模态反诈引擎测试 ===")
    
    # 准备测试用例
    test_cases = [
        "中奖了！恭喜您获得一等奖，请点击链接领取，并汇款手续费到账户XXX",
        "您好，我是您的朋友，最近家里出了点急事，能不能借我2000元，下午就还你。",
        "【工商银行】您的网银电子密码器即将过期，请登录我行手机网查看。",
        "明天开会记得带上项目计划书，谢谢。"
    ]
    
    # 初始化引擎
    engine = FraudDetectionEngine(config)
    
    # 是否通过命令行参数决定是否测试 Qwen3 (因为加载慢)
    use_qwen3 = "--qwen3" in sys.argv
    
    for i, text in enumerate(test_cases):
        print(f"\n[测试用例 {i+1}]: {text[:50]}...")
        
        start_time = time.time()
        results = engine.detect(text, use_qwen3=use_qwen3)
        end_time = time.time()
        
        print(f"检测结果: {'诈骗' if results['is_fraud'] else '正常'}")
        if results['is_fraud']:
            print(f"触发模型: {', '.join(results['trigger_models'])}")
        
        print("详细得分:")
        for name, detail in results['details'].items():
            print(f"  - {name}: 概率={detail['prob']:.4f}, 预测类型={detail['type']}")
            
        print(f"耗时: {end_time - start_time:.2f}秒")

if __name__ == "__main__":
    test_inference()
