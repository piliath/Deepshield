document.addEventListener('DOMContentLoaded', () => {
    const reportsList = document.getElementById('reportsList');
    const modalOverlay = document.getElementById('reportDetailModal');
    const closeDetailBtn = document.getElementById('closeDetailBtn');
    const sortByTimeBtn = document.getElementById('sortByTime');
    const sortByCategoryBtn = document.getElementById('sortByCategory');
    
    let originalReports = [];

    // 加载报告列表
    async function loadReports() {
        try {
            const response = await fetch('/api/reports/');
            const reports = await response.json();
            originalReports = reports;
            renderReports(reports);
        } catch (error) {
            reportsList.innerHTML = '<div style="color:red; text-align:center; padding: 2rem;">加载拦截报告失败</div>';
        }
    }

    // 渲染卡片
    function renderReports(reports) {
        if (!reports || reports.length === 0) {
            reportsList.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1; text-align:center; padding: 4rem; opacity: 0.6;">
                    <h3>当前暂无拦截记录</h3>
                    <p>当系统扫描并拦截诈骗邮件后，报告将在此呈现。</p>
                </div>
            `;
            return;
        }

        reportsList.innerHTML = reports.map(r => {
            const data = r.data || {};
            const timeStr = new Date(r.timestamp).toLocaleString();
            return `
                <div class="card clickable-card" onclick="viewReport('${r.id}')" style="cursor: pointer; position: relative;">
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.5rem;">${timeStr}</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                        <h4 style="font-size: 1rem;">${data.fraud_type || '疑似诈骗'}</h4>
                        <span class="mailbox-status active" style="font-size: 0.7rem; color: #FFF; background: var(--danger-color);">高风险</span>
                    </div>
                    <p style="font-size: 0.85rem; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; margin-bottom: 1rem;">
                        ${data.raw_content_summary || '邮件内容摘要加载中...'}
                    </p>
                    <div style="font-size: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: var(--primary-color);">来自: ${r.mailbox}</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    // 按时间排序
    function sortByTime() {
        const sortedReports = [...originalReports].sort((a, b) => {
            return new Date(b.timestamp) - new Date(a.timestamp);
        });
        renderReports(sortedReports);
    }

    // 按类别排序
    function sortByCategory() {
        const sortedReports = [...originalReports].sort((a, b) => {
            const categoryA = (a.data && a.data.fraud_type) ? a.data.fraud_type : '疑似诈骗';
            const categoryB = (b.data && b.data.fraud_type) ? b.data.fraud_type : '疑似诈骗';
            return categoryA.localeCompare(categoryB);
        });
        renderReports(sortedReports);
    }

    // 查看报告详情
    window.viewReport = async (reportId) => {
        modalOverlay.style.display = 'flex';
        const aiAnalysis = document.getElementById('aiAnalysis');
        const emailSummary = document.getElementById('emailSummary');
        const modelScores = document.getElementById('modelScores');
        const suggest = document.getElementById('suggest');
        
        aiAnalysis.innerHTML = '深度分析中...';
        modelScores.innerHTML = '加载分值中...';
        
        try {
            const response = await fetch(`/api/reports/${reportId}`);
            const report = await response.json();
            const data = report.data || {};
            const detection = report.detection || { details: {} };

            document.getElementById('detailSubTitle').innerText = `报告 ID: ${report.id} | 拦截于: ${new Date(report.timestamp).toLocaleString()}`;
            
            aiAnalysis.innerHTML = data.user_alert_content || "模型未返回详细分析内容。";
            emailSummary.innerHTML = `
                <p><strong>邮箱:</strong> ${report.mailbox}</p>
                <p><strong>概要:</strong> ${data.raw_content_summary || '未抓取到摘要'}</p>
                <p style="margin-top: 0.5rem; font-style: italic; font-size: 0.8rem;">(已隐藏原始代码/链接以防误触)</p>
            `;
            
            // 渲染各小模型分值
            let scoresHtml = '';
            for (const [model, detail] of Object.entries(detection.details)) {
                scoresHtml += `
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.875rem;">
                        <span style="color: #475569;">${model}</span>
                        <span style="font-weight: 700; color: ${detail.prob > 0.85 ? 'var(--danger-color)' : 'var(--success-color)'}">${(detail.prob * 100).toFixed(2)}%</span>
                    </div>
                `;
            }
            modelScores.innerHTML = scoresHtml || "未记录到分值";
            
            suggest.innerHTML = `
                <p style="padding: 1rem;">${data.similar_cases_suggest || '请勿点击任何链接，不要回复邮件。如有资金往来，请立即报警。'}</p>
            `;

        } catch (error) {
            aiAnalysis.innerHTML = '加载详情失败';
        }
    }

    closeDetailBtn.addEventListener('click', () => {
        modalOverlay.style.display = 'none';
    });

    sortByTimeBtn.addEventListener('click', sortByTime);
    sortByCategoryBtn.addEventListener('click', sortByCategory);

    loadReports();
});
