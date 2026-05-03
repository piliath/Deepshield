from flask import Blueprint, request, jsonify
from config import load_json, save_json, MAILBOXES_FILE, STATS_FILE
from services.email_fetcher import EmailFetcher
import os
from datetime import datetime

mailbox_bp = Blueprint('mailbox', __name__)
fetcher = EmailFetcher()

@mailbox_bp.route('/<email>/check', methods=['POST'])
def check_mailbox(email):
    mailboxes = load_json(MAILBOXES_FILE)
    target = next((m for m in mailboxes if m['email'] == email), None)
    
    if not target:
        return jsonify({"success": False, "message": "未找到该邮箱"}), 404
    
    # 获取未读邮件
    new_emails = fetcher.fetch_unseen_emails(target['email'], target['auth_code'])
    
    # 更新最后检查时间
    for m in mailboxes:
        if m['email'] == email:
            m['last_checked'] = datetime.now().isoformat()
            break
            
    save_json(MAILBOXES_FILE, mailboxes)
    
    return jsonify({
        "success": True, 
        "new_count": len(new_emails),
        "emails": [
            {"subject": e['subject'], "from": e['from'], "content_len": len(e['body'])} 
            for e in new_emails
        ]
    })

@mailbox_bp.route('/', methods=['GET'])
def get_mailboxes():
    mailboxes = load_json(MAILBOXES_FILE)
    return jsonify(mailboxes)

@mailbox_bp.route('/', methods=['POST'])
def add_mailbox():
    data = request.json
    email = data.get('email')
    auth_code = data.get('auth_code')
    
    if not email or not auth_code:
        return jsonify({"success": False, "message": "邮箱或授权码不能为空"}), 400
    
    mailboxes = load_json(MAILBOXES_FILE)
    # 检查是否已存在
    if any(m['email'] == email for m in mailboxes):
        return jsonify({"success": False, "message": "该邮箱已存在"}), 400
    
    new_mailbox = {
        "email": email,
        "auth_code": auth_code,
        "added_at": datetime.now().isoformat(),
        "last_checked": None,
        "status": "active",
        "intercept_count": 0
    }
    
    mailboxes.append(new_mailbox)
    save_json(MAILBOXES_FILE, mailboxes)
    return jsonify({"success": True, "mailbox": new_mailbox})

@mailbox_bp.route('/<email>', methods=['DELETE'])
def delete_mailbox(email):
    mailboxes = load_json(MAILBOXES_FILE)
    new_mailboxes = [m for m in mailboxes if m['email'] != email]
    
    if len(new_mailboxes) == len(mailboxes):
        return jsonify({"success": False, "message": "未找到该邮箱"}), 404
    
    save_json(MAILBOXES_FILE, new_mailboxes)
    return jsonify({"success": True})
