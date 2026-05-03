from flask import Blueprint, jsonify
from config import load_json, STATS_FILE, MAILBOXES_FILE
import os

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/stats', methods=['GET'])
def get_stats():
    stats = load_json(STATS_FILE, default={
        "total_intercepted": 0,
        "daily_counts": {},
        "type_distribution": {
            "公检法诈骗": 0,
            "贷款诈骗": 0,
            "客服诈骗": 0,
            "熟人诈骗": 0
        }
    })
    
    mailboxes = load_json(MAILBOXES_FILE)
    stats["mailbox_count"] = len(mailboxes)
    
    return jsonify(stats)
