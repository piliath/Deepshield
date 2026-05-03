import sys
import os

# 强制添加项目根目录到 sys.path，保证能 import config
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print(f"[Debug] Root dir: {root_dir}")

try:
    import imaplib
    import socket # Added socket import
    import email
    from email.header import decode_header
    import config
    from datetime import datetime
    print("[Debug] Imports successful")
except Exception as e:
    print(f"[Error] Import failed: {str(e)}")
    sys.exit(1)

class EmailFetcher:
    def __init__(self):
        self.imap_host = config.IMAP_HOST
        self.imap_port = config.IMAP_PORT
        print(f"[Debug] Fetcher initialized with {self.imap_host}:{self.imap_port}")

    def _get_ipv4_host(self, host):
        """尝试获取 IPv4 地址以避开 IPv6 连接问题"""
        try:
            addr_info = socket.getaddrinfo(host, None, socket.AF_INET)
            if addr_info:
                return addr_info[0][4][0]
        except:
            pass
        return host

    def _decode_str(self, s):
        if s is None:
            return ""
        try:
            decoded_list = decode_header(s)
            res = []
            for content, charset in decoded_list:
                if isinstance(content, bytes):
                    res.append(content.decode(charset or 'utf-8', errors='ignore'))
                else:
                    res.append(str(content))
            return "".join(res)
        except:
            return str(s)

    def _get_email_body(self, msg):
        """解析邮件正文内容 (优先 Plain Text，无则使用 HTML)"""
        body = ""
        html_body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if "attachment" in content_disposition:
                    continue
                
                try:
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue
                    charset = part.get_content_charset() or 'utf-8'
                    text_content = payload.decode(charset, errors='ignore')
                    
                    if content_type == "text/plain":
                        body = text_content
                        break # 找到了纯文本，直接退出循环
                    elif content_type == "text/html":
                        html_body = text_content
                except:
                    continue
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or 'utf-8'
                    body = payload.decode(charset, errors='ignore')
            except:
                pass

        # 如果没有 Plain Text，就用 HTML
        return body if body.strip() else html_body

    def fetch_unseen_emails(self, email_address, auth_code):
        results = []
        try:
            # 强制通过 IPv4 连接网易 IMAP（避开某些网络环境下解析 AAAA 却无法连接的问题）
            ipv4_host = self._get_ipv4_host(self.imap_host)
            print(f"[{email_address}] 正在通过 {ipv4_host} 连接 {self.imap_host}...")
            mail = imaplib.IMAP4_SSL(ipv4_host, self.imap_port)
            mail.login(email_address, auth_code)
            print(f"[{email_address}] 登录成功")
            
            # 网易 163 特殊要求：在 SELECT 之前发送 ID 命令，否则会报错 "Unsafe Login"
            try:
                # 声明客户端身份
                mail.xatom('ID', '("name" "anticheat-client" "version" "1.0.0" "vendor" "my-company")')
                print(f"[{email_address}] 已发送 ID 命令")
            except Exception as e:
                print(f"[{email_address}] 发送 ID 命令失败 (可能不支持): {str(e)}")
            
            # 选择收件箱
            status, data = mail.select("INBOX", readonly=False)
            if status != 'OK':
                print(f"[{email_address}] 无法选择收件箱 (INBOX): {status}, {data}")
                return []
            
            print(f"[{email_address}] 已选择收件箱，准备搜索...")
            status, response = mail.search(None, 'UNSEEN')
            
            if status != 'OK':
                print(f"[{email_address}] 搜索未读邮件失败")
                return []

            email_ids = response[0].split()
            print(f"[{email_address}] 发现 {len(email_ids)} 封新邮件")

            for e_id in email_ids:
                res, data = mail.fetch(e_id, '(RFC822)')
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)

                subject = self._decode_str(msg.get("Subject"))
                sender = self._decode_str(msg.get("From"))
                date_str = msg.get("Date")
                body = self._get_email_body(msg)

                results.append({
                    "uid": e_id.decode(),
                    "mailbox": email_address,
                    "from": sender,
                    "subject": subject,
                    "body": body,
                    "date": date_str
                })
                print(f"[{email_address}] 已抓取: {subject}")

            mail.close()
            mail.logout()
        except Exception as e:
            print(f"[{email_address}] 拉取异常: {str(e)}")
            
        return results

if __name__ == "__main__":
    from config import load_json, MAILBOXES_FILE
    
    print(f"[Debug] Loading mailboxes from {MAILBOXES_FILE}")
    fetcher = EmailFetcher()
    mailboxes = load_json(MAILBOXES_FILE)
    
    if not mailboxes:
        print("未发现监控邮箱，请先在界面添加。")
    else:
        print(f"找到 {len(mailboxes)} 个邮箱，开始检查...")
        for m in mailboxes:
            print(f"\n--- 正在检查: {m['email']} ---")
            emails = fetcher.fetch_unseen_emails(m['email'], m['auth_code'])
            if not emails:
                print("未发现新邮件。")
            else:
                for e in emails:
                    print(f"主题: {e['subject']}")
                    print(f"发件人: {e['from']}")
                    print(f"内容长度: {len(e['body'])}")
                    print("-" * 20)
