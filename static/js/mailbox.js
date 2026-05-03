document.addEventListener('DOMContentLoaded', () => {
    const mailboxList = document.getElementById('mailboxList');
    const modalOverlay = document.getElementById('modalOverlay');
    const openModalBtn = document.getElementById('openModalBtn');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const addMailboxForm = document.getElementById('addMailboxForm');

    // 加载邮箱列表
    async function loadMailboxes() {
        mailboxList.innerHTML = '<div style="text-align:center; padding: 2rem;">正在加载中...</div>';
        try {
            const response = await fetch('/api/mailboxes/');
            const mailboxes = await response.json();
            renderMailboxes(mailboxes);
        } catch (error) {
            mailboxList.innerHTML = '<div style="color:var(--danger-color); text-align:center; padding: 2rem;">加载失败，请重试</div>';
        }
    }

    // 渲染邮箱卡片
    function renderMailboxes(mailboxes) {
        if (mailboxes.length === 0) {
            mailboxList.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1; text-align:center; padding: 4rem; background: white; border-radius: 1rem; border: 2px dashed var(--border-color);">
                    <img src="/static/icon/email.svg" style="width: 48px; height: 48px; opacity: 0.5; margin-bottom: 1rem;">
                    <h3 style="color: #64748B;">暂无监控邮箱</h3>
                    <p style="color: #94A3B8; margin-top: 0.5rem;">点击右下角按钮添加第一个监控邮箱</p>
                </div>
            `;
            return;
        }

        mailboxList.innerHTML = mailboxes.map(m => `
            <div class="card mailbox-card">
                <div class="mailbox-header">
                    <div class="mailbox-info">
                        <h3>${m.email}</h3>
                        <span class="mailbox-status ${m.status}">${m.status === 'active' ? '● 监控中' : '○ 已停止'}</span>
                    </div>
                </div>
                <div class="mailbox-stats">
                    <div class="stat-item">
                        <span class="stat-label">拦截总量</span>
                        <span class="stat-value">${m.intercept_count || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">上次同步</span>
                        <span class="stat-value" style="font-size: 0.75rem; color: var(--text-secondary);">
                            ${m.last_checked ? new Date(m.last_checked).toLocaleString() : '从未'}
                        </span>
                    </div>
                </div>
                <div class="mailbox-actions">
                    <button class="btn btn-primary" onclick="syncMailbox('${m.email}')" id="sync-${m.email.replace('@','-').replace('.','-')}">同步</button>
                    <button class="btn btn-danger" onclick="deleteMailbox('${m.email}')">删除</button>
                </div>
            </div>
        `).join('');
    }

    // 同步邮件功能
    window.syncMailbox = async (email) => {
        const btnId = `sync-${email.replace('@','-').replace('.','-')}`;
        const btn = document.getElementById(btnId);
        const originalText = btn.innerText;
        
        btn.disabled = true;
        btn.innerText = '同步中...';
        
        try {
            const response = await fetch(`/api/mailboxes/${email}/check`, { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                alert(`同步完成！发现 ${result.new_count} 封新邮件。`);
                loadMailboxes(); // 刷新列表以更新最后同步时间
            } else {
                alert(result.message || '同步失败');
            }
        } catch (error) {
            alert('服务器通讯失败');
        } finally {
            btn.disabled = false;
            btn.innerText = originalText;
        }
    };

    // 模态框逻辑
    openModalBtn.addEventListener('click', () => {
        modalOverlay.style.display = 'flex';
    });

    const closeModal = () => {
        modalOverlay.style.display = 'none';
        addMailboxForm.reset();
    };

    closeModalBtn.addEventListener('click', closeModal);
    cancelBtn.addEventListener('click', closeModal);

    // 提交添加邮箱表单
    addMailboxForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('email').value;
        const authCode = document.getElementById('authCode').value;

        try {
            const response = await fetch('/api/mailboxes/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, auth_code: authCode })
            });

            const result = await response.json();
            if (result.success) {
                closeModal();
                loadMailboxes();
                alert('添加成功！');
            } else {
                alert(result.message || '添加失败');
            }
        } catch (error) {
            alert('服务器通讯失败');
        }
    });

    // 删除邮箱功能
    window.deleteMailbox = async (email) => {
        if (!confirm(`确定要移除对邮箱 ${email} 的监控吗？相关拦截记录将保留，但不再接收新邮件。`)) return;

        try {
            const response = await fetch(`/api/mailboxes/${email}`, { method: 'DELETE' });
            const result = await response.json();
            if (result.success) {
                loadMailboxes();
            } else {
                alert(result.message || '删除失败');
            }
        } catch (error) {
            alert('删除操作失败');
        }
    };

    loadMailboxes();
});
