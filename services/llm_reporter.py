import json
import os
from openai import OpenAI
import config
from datetime import datetime

class LLMReporter:
    def __init__(self, cfg):
        self.cfg = cfg
        # 设置 OpenAI 兼容的 Base URL (指向 DashScope)
        self.client = OpenAI(
            api_key=cfg.load_settings().get("dashscope_api_key"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def generate_report(self, email_data, detection_results):
        """
        生成两份报告：用户告警信 和 系统存档报告
        """
        prompt = f"""
你是一个专业的反诈系统分析专家。
系统刚刚拦截了一封疑似诈骗邮件，信息如下：
- 用户邮箱: {email_data.get('mailbox')}
- 发件人: {email_data.get('from')}
- 邮件主题: {email_data.get('subject')}
- 邮件正文: {email_data.get('body')}
- 拦截时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

小模型检测结果（OR逻辑已触发拦截）：
{json.dumps(detection_results, indent=2, ensure_ascii=False)}

请根据以上信息，输出两部分内容。

### [用户告警邮件内容]
- 包含：拦截时间、发送时间（{email_data.get('date')}）、可能诈骗内容摘要。
- 重点：以通俗易懂的方式向用户解释“为什么这是诈骗”（结合小模型的类型判定，如贷款诈骗、冒充公检法等）。

### [系统存档 JSON 报告]
请按以下 JSON 结构输出系统报告：
{{
  "intercept_time": "...",
  "user_mailbox": "...",
  "raw_content_summary": "...",
  "fraud_type": "...",
  "confidence_avg": "...",
  "user_alert_content": "...",
  "similar_cases_suggest": "..."
}}

注意：请直接输出内容，JSON 部分要包含在对应的 markdown 代码块中。
"""

        try:
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {'role': 'system', 'content': '你是一个严谨的反诈分析师，擅长识破各类网络电信诈骗。'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            content = completion.choices[0].message.content
            
            # 解析出两部分
            alert_mail = ""
            json_report = {}
            
            # 简单分割（实际可根据更强的标记分割）
            if "### [用户告警邮件内容]" in content:
                alert_mail = content.split("### [用户告警邮件内容]")[1].split("### [系统存档 JSON 报告]")[0].strip()
            
            # 提取 JSON 块
            if "```json" in content:
                json_raw = content.split("```json")[1].split("```")[0].strip()
                json_report = json.loads(json_raw)
            else:
                # 备选方案
                json_report = {"content": content}

            return alert_mail, json_report
            
        except Exception as e:
            print(f"[LLM] 生成报告失败: {str(e)}")
            return "分析服务暂时不可用，请手动核查。", {"error": str(e)}

if __name__ == "__main__":
    # 测试代码 (需要 settings.json 中有有效 API Key)
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    reporter = LLMReporter(config)
    
    test_mail = {
        "mailbox": "lihuahua@163.com",
        "from": "bank_service@bank.com",
        "subject": "您的银行账户已冻结，请点击链接解冻",
        "body": "亲爱的用户，您的工商银行账户由于涉嫌异常交易被临时冻结。请点击 http://fake-bank.com/reset 进行身份验证，否则账户将被永久注销。",
        "date": "2024-03-24"
    }
    
    test_results = {
        "is_fraud": True,
        "trigger_models": ["LightGBM", "BERT"],
        "details": {
            "LightGBM": {"prob": 0.98, "type": "贷款诈骗"},
            "BERT": {"prob": 0.95, "type": "贷款诈骗"}
        }
    }
    
    print("正在测试 Qwen-Plus 报告生成...")
    mail, report = reporter.generate_report(test_mail, test_results)
    
    print("\n[生成的告警邮件]:")
    print(mail)
    print("\n[生成的存档 JSON]:")
    print(json.dumps(report, indent=4, ensure_ascii=False))
