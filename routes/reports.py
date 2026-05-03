from flask import Blueprint, jsonify, request
from config import load_json, REPORTS_FILE
import os

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/', methods=['GET'])
def get_reports():
    # 读取所有拦截报告
    reports = load_json(REPORTS_FILE)
    return jsonify(reports)

@reports_bp.route('/<report_id>', methods=['GET'])
def get_report_detail(report_id):
    reports = load_json(REPORTS_FILE)
    report = next((r for r in reports if r['id'] == report_id), None)
    
    if not report:
        return jsonify({"success": False, "message": "未找到报告"}), 404
        
    return jsonify(report)
