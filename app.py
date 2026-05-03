from flask import Flask, render_template, request, jsonify
from config import load_settings, save_settings
import os
import config
from services.scheduler import AnticheatScheduler

app = Flask(__name__)

# 全局初始化调度器
scheduler = AnticheatScheduler(config)

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/mailbox")
def mailbox():
    return render_template("mailbox.html")

@app.route("/reports")
def reports_list():
    return render_template("reports.html")

@app.route("/sms-input")
def sms_input():
    return render_template("sms_input.html")

@app.route("/settings")
def settings_page():
    return render_template("settings.html")

# 系统设置 API
@app.get("/api/settings")
def get_settings_api():
    return jsonify(load_settings())

@app.post("/api/settings")
def save_settings_api():
    new_settings = request.json
    save_settings(new_settings)
    return jsonify({"success": True})

# 引入蓝图（后续添加）
from routes.mailbox import mailbox_bp
from routes.dashboard import dashboard_bp
from routes.reports import reports_bp
from routes.sms import sms_bp
app.register_blueprint(mailbox_bp, url_prefix='/api/mailboxes')
app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
app.register_blueprint(reports_bp, url_prefix='/api/reports')
app.register_blueprint(sms_bp, url_prefix='/api/sms')

if __name__ == "__main__":
    # 启动后台监控
    scheduler.start()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) # 禁用热重载以防重复启动线程
