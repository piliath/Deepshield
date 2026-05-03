from flask import Blueprint, request, jsonify
import csv
import os
from datetime import datetime

sms_bp = Blueprint('sms', __name__)
CSV_PATH = os.path.join("data", "sms_records.csv")

@sms_bp.route('/save', methods=['POST'])
def save_sms():
    data = request.json
    text = data.get('text', "").replace("\n", " ").replace(",", "，")
    label = data.get('label', "正常")
    
    if not text:
        return jsonify({"success": False, "message": "正文不能为空"}), 400

    # 写入 CSV
    file_exists = os.path.isfile(CSV_PATH)
    try:
        with open(CSV_PATH, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['content', 'label', 'timestamp'])
            writer.writerow([text, label, datetime.now().isoformat()])
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
