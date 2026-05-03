import time
import json
import os
from datetime import datetime
from threading import Thread
import config
from services.email_fetcher import EmailFetcher
from services.model_engine import FraudDetectionEngine
from services.llm_reporter import LLMReporter

class AnticheatScheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fetcher = EmailFetcher()
        self.engine = FraudDetectionEngine(cfg)
        self.reporter = LLMReporter(cfg)
        self.running = False

    def _save_report_to_file(self, mailbox, result_report, detection_results):
        """将拦截报告保存到本地 data/reports.json"""
        path = self.cfg.REPORTS_FILE
        reports = self.cfg.load_json(path)
        
        # 补充额外信息
        report_entry = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "mailbox": mailbox,
            "status": "intercepted",
            "timestamp": datetime.now().isoformat(),
            "detection": detection_results, # 存储小模型结果
            "data": result_report
        }
        
        reports.insert(0, report_entry) # 最新在最前
        self.cfg.save_json(path, reports)
        
        # 更新全站统计 (总拦截数)
        stats = self.cfg.load_json(self.cfg.STATS_FILE, default={
            "total_intercepted": 0, "daily_counts": {}, "type_distribution": {}
        })
        stats["total_intercepted"] += 1
        
        today = datetime.now().strftime("%Y-%m-%d")
        stats["daily_counts"][today] = stats["daily_counts"].get(today, 0) + 1
        
        type_name = result_report.get("fraud_type", "未知诈骗")
        stats["type_distribution"][type_name] = stats["type_distribution"].get(type_name, 0) + 1
        
        self.cfg.save_json(self.cfg.STATS_FILE, stats)

    def _update_mailbox_stats(self, email_address):
        """更新邮箱的拦截总计数"""
        mailboxes = self.cfg.load_json(self.cfg.MAILBOXES_FILE)
        for m in mailboxes:
            if m['email'] == email_address:
                m['intercept_count'] = m.get('intercept_count', 0) + 1
                m['last_checked'] = datetime.now().isoformat()
                break
        self.cfg.save_json(self.cfg.MAILBOXES_FILE, mailboxes)

    def process_all_mailboxes(self):
        """执行一轮全量扫描"""
        settings = self.cfg.load_settings()
        mailboxes = self.cfg.load_json(self.cfg.MAILBOXES_FILE)
        
        if not mailboxes:
            print("[Scheduler] 暂无监控邮箱，跳过扫描")
            return

        use_qwen3 = settings.get("use_qwen3_model", False)
        print(f"[Scheduler] 开始扫描 {len(mailboxes)} 个邮箱... (Qwen3开启: {use_qwen3})")

        for m in mailboxes:
            try:
                # 1. 抓取新邮件
                new_emails = self.fetcher.fetch_unseen_emails(m['email'], m['auth_code'])
                if not new_emails:
                    continue
                
                print(f"[Scheduler] 发现 {len(new_emails)} 封新邮件，正在判定其风险...")
                
                for mail in new_emails:
                    # 2. 小模型判定 (OR 逻辑)
                    detection = self.engine.detect(mail['body'], use_qwen3=use_qwen3)
                    
                    if detection["is_fraud"]:
                        print(f"[Scheduler] !!! 检出诈骗邮件: {mail['subject']} !!!")
                        
                        # 3. 大模型撰写分析报告
                        alert_mail, result_report = self.reporter.generate_report(mail, detection)
                        
                        # 4. 存档报告
                        self._save_report_to_file(m['email'], result_report, detection)
                        
                        # 5. 更新计数
                        self._update_mailbox_stats(m['email'])
                        
                        # (TODO) 6. 以后这里可以调用 SMTP 为用户发送告警信
                        print(f"[Scheduler] 报告已存档到 data/reports.json")
                    else:
                        print(f"[Scheduler] 安全邮件: {mail['subject']}")
                        
            except Exception as e:
                print(f"[Scheduler] 处理邮箱 {m['email']} 时发生致命错误: {str(e)}")

    def run_loop(self):
        """主循环，按频率执行"""
        self.running = True
        while self.running:
            try:
                self.process_all_mailboxes()
                
                settings = self.cfg.load_settings()
                interval = int(settings.get("fetch_interval", 300)) # 默认 5 分钟
                
                print(f"[Scheduler] 扫描完成，{interval} 秒后进行下一轮检测...")
                
                # 动态休眠
                start_sleep = time.time()
                while time.time() - start_sleep < interval:
                    if not self.running: break
                    time.sleep(1)
                    
            except Exception as e:
                print(f"[Scheduler] 扫描循环崩溃: {str(e)}")
                time.sleep(60)

    def start(self):
        """在后台线程开启调度"""
        if not self.running:
            thread = Thread(target=self.run_loop, daemon=True)
            thread.start()
            print("[Scheduler] 后台扫描线程已启动")

    def stop(self):
        self.running = False
        print("[Scheduler] 定时扫描已停止")
