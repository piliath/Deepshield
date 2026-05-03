document.addEventListener('DOMContentLoaded', () => {
    // 加载总统计数据
    async function loadStats() {
        try {
            const statsResp = await fetch('/api/dashboard/stats');
            const stats = await statsResp.json();
            
            document.getElementById('totalIntercepted').innerText = stats.total_intercepted || 0;
            
            const today = new Date().toISOString().split('T')[0];
            document.getElementById('todayIntercepted').innerText = stats.daily_counts[today] || 0;
            document.getElementById('mailboxCount').innerText = stats.mailbox_count || 0;
            
            // 计算活跃诈骗 (出现次数最多的类型)
            const typeDist = stats.type_distribution || {};
            let maxType = "暂无活跃类型";
            let maxCount = 0;
            let total = 0;
            for (const [type, count] of Object.entries(typeDist)) {
                total += count;
                if (count > maxCount) {
                    maxCount = count;
                    maxType = type;
                }
            }
            if (total > 0) {
                document.getElementById('activeFraudType').innerText = maxType;
                document.getElementById('activeFraudProb').innerText = `占比: ${((maxCount / total) * 100).toFixed(1)}%`;
            }

            // 加载最近拦截列表 (取前5条)
            const reportsResp = await fetch('/api/reports/');
            const reports = await reportsResp.json();
            renderRecentTable(reports.slice(0, 5));

            // 渲染图表
            renderCharts(stats);
        } catch (error) {
            console.error('加载统计数据失败', error);
        }
    }

    function renderRecentTable(reports) {
        const tbody = document.getElementById('recentList');
        if (!reports || reports.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:2rem;">暂无拦截记录</td></tr>';
            return;
        }

        tbody.innerHTML = reports.map(r => {
            const data = r.data || {};
            const detection = r.detection || { details: {} };
            // 获取最高分模型
            let maxModel = "AI检测";
            let maxProb = 0;
            for(const [m, d] of Object.entries(detection.details)) {
                if(d.prob > maxProb) { maxProb = d.prob; maxModel = m; }
            }

            return `
                <tr style="border-bottom: 1px solid var(--border-color);">
                    <td style="padding: 1rem 0;">${new Date(r.timestamp).toLocaleString()}</td>
                    <td>${r.mailbox}</td>
                    <td>${data.fraud_type || '疑似诈骗'}</td>
                    <td><span class="mailbox-status active" style="background:#FEE2E2; color:#B91C1C;">${maxModel}</span></td>
                    <td>${(maxProb * 100).toFixed(1)}%</td>
                    <td style="text-align: center;"><a href="/reports" style="color:var(--primary-color)">详情</a></td>
                </tr>
            `;
        }).join('');
    }

    function renderCharts(stats) {
        const trendCtx = document.getElementById('trendChart').getContext('2d');
        
        // 生成近 7 天刻度
        const dates = [];
        const counts = [];
        for (let i = 6; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const dateStr = d.toISOString().split('T')[0];
            dates.push(dateStr.slice(5)); // 只显 MM-DD
            counts.push(stats.daily_counts[dateStr] || 0);
        }

        new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: '每日拦截量',
                    data: counts,
                    borderColor: '#4F46E5',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } }
            }
        });

        // 分布图
        const typeCtx = document.getElementById('typeChart').getContext('2d');
        const typeData = stats.type_distribution || {};
        new Chart(typeCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(typeData),
                datasets: [{
                    data: Object.values(typeData),
                    backgroundColor: ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#94A3B8'],
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 12 } }
                }
            }
        });
    }

    loadStats();
});
