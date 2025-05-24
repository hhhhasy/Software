import session from '../utils/session.js';
import { showLoading as showGlobalLoading, hideLoading as hideGlobalLoading, showError as showGlobalError, showSuccess as showGlobalSuccess } from '../utils/uiUtils.js';

/**
 * 维护人员界面逻辑
 */

// 配置
const API_URL = 'http://localhost:8000/api';
const LOGS_PER_PAGE = 50;

// 状态变量
let currentLogPage = 1;
let totalLogPages = 1;
let currentLogLevel = '';
let currentAction = null;

// DOM 元素引用
const logContainer = document.getElementById('logContainer');
const logLevelFilter = document.getElementById('logLevelFilter');
const refreshLogsBtn = document.getElementById('refreshLogs');
const prevLogsBtn = document.getElementById('prevLogs');
const nextLogsBtn = document.getElementById('nextLogs');
const pageInfo = document.getElementById('pageInfo');
const maintainerUser = document.getElementById('maintainerUser');
const maintainerLogout = document.getElementById('logout');
const actionModal = document.getElementById('actionModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const confirmAction = document.getElementById('confirmAction');
const cancelAction = document.getElementById('cancelAction');
const closeModal = document.querySelector('.close-modal');

// 系统状态元素
const serverStatus = document.getElementById('serverStatus');
const cpuUsage = document.getElementById('cpuUsage');
const memoryUsage = document.getElementById('memoryUsage');
const diskUsage = document.getElementById('diskUsage');

// 维护工具按钮
const cleanCacheBtn = document.getElementById('cleanCacheBtn');
const dbMaintenanceBtn = document.getElementById('dbMaintenanceBtn');
const restartServicesBtn = document.getElementById('restartServicesBtn');
const securityScanBtn = document.getElementById('securityScanBtn');

/**
 * 初始化页面
 */
function init() {
    // 统一会话管理
    const currentUser = session.get('currentUser');
    maintainerUser.textContent = currentUser ? currentUser.username : '未登录';

    // 加载日志
    loadLogs();

    // 模拟系统状态指标更新
    updateSystemStatus();

    // 绑定事件监听
    bindEvents();
}

/**
 * 绑定事件监听
 */
function bindEvents() {
    // 日志相关事件
    logLevelFilter.addEventListener('change', () => {
        currentLogLevel = logLevelFilter.value;
        currentLogPage = 1;
        loadLogs();
    });

    refreshLogsBtn.addEventListener('click', () => {
        loadLogs();
    });

    prevLogsBtn.addEventListener('click', () => {
        if (currentLogPage > 1) {
            currentLogPage--;
            loadLogs();
        }
    });

    nextLogsBtn.addEventListener('click', () => {
        if (currentLogPage < totalLogPages) {
            currentLogPage++;
            loadLogs();
        }
    });

    // 修改登出逻辑
    maintainerLogout.addEventListener('click', () => {
        session.clear();
        window.location.replace('login.html');
    });

    // 模态框事件
    closeModal.addEventListener('click', hideModal);
    cancelAction.addEventListener('click', hideModal);
    confirmAction.addEventListener('click', executeAction);

    // 维护工具按钮事件
    cleanCacheBtn.addEventListener('click', () => showConfirmation('清理缓存', '确定要清理系统缓存？这将删除所有临时文件。', 'cleanCache'));
    dbMaintenanceBtn.addEventListener('click', () => showConfirmation('数据库维护', '确定要执行数据库维护？此操作可能需要几分钟时间。', 'dbMaintenance'));
    restartServicesBtn.addEventListener('click', () => showConfirmation('重启服务', '确定要重启系统服务？这将暂时中断服务。', 'restartServices'));
    securityScanBtn.addEventListener('click', () => showConfirmation('安全扫描', '确定要执行系统安全扫描？', 'securityScan'));
}

/**
 * 加载系统日志
 */
async function loadLogs() {
    try {
        showGlobalLoading('日志加载中...');
        
        // 构建请求参数
        const requestData = {
            limit: LOGS_PER_PAGE
        };
        
        // 如果选择了日志级别，添加到请求参数
        if (currentLogLevel) {
            requestData.level = currentLogLevel;
        }
        
        // 发送POST请求
        const response = await fetch(`${API_URL}/logs`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error('无法获取日志数据');
        }
        
        const data = await response.json();
        
        // 更新分页信息
        totalLogPages = Math.ceil(data.total_entries / LOGS_PER_PAGE);
        pageInfo.textContent = `第 ${currentLogPage} 页 / 共 ${totalLogPages || 1} 页`;
        
        // 更新分页按钮状态
        prevLogsBtn.disabled = currentLogPage <= 1;
        nextLogsBtn.disabled = currentLogPage >= totalLogPages;
        
        // 清空并重新填充日志容器
        logContainer.innerHTML = '';
        
        if (!data.logs || data.logs.length === 0) {
            logContainer.innerHTML = '<div class="empty-logs">没有找到日志记录</div>';
            return;
        }
        
        // 渲染日志项
        data.logs.forEach(log => {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            
            // 调整日志显示布局，减小级别占用的空间
            logEntry.innerHTML = `
                <div class="log-timestamp">${log.timestamp}</div>
                <div class="log-level ${log.level}">${log.level}</div>
                <div class="log-message">${log.message}</div>
                <div class="log-expand"><i class="fas fa-expand-alt"></i></div>
            `;
            
            // 点击展开日志详情
            logEntry.querySelector('.log-expand').addEventListener('click', () => {
                showLogDetail(log);
            });
            
            logContainer.appendChild(logEntry);
        });
    } catch (error) {
        showGlobalError('加载日志失败: ' + error.message);
        logContainer.innerHTML = `<div class="error-message">加载日志失败: ${error.message}</div>`;
    } finally {
        hideGlobalLoading();
    }
}

/**
 * 显示日志详情
 */
function showLogDetail(log) {
    modalTitle.textContent = '日志详情';
    
    // 格式化日志时间，使其更易读
    const timestamp = log.timestamp;
    
    // 根据日志级别设置不同的图标和颜色
    let levelIcon = '';
    switch(log.level) {
        case 'INFO':
            levelIcon = '<i class="fas fa-info-circle" style="color: var(--primary-color)"></i>';
            break;
        case 'WARNING':
            levelIcon = '<i class="fas fa-exclamation-triangle" style="color: var(--warning-color)"></i>';
            break;
        case 'ERROR':
            levelIcon = '<i class="fas fa-times-circle" style="color: var(--error-color)"></i>';
            break;
        case 'DEBUG':
            levelIcon = '<i class="fas fa-bug" style="color: #93979f"></i>';
            break;
        case 'CRITICAL':
            levelIcon = '<i class="fas fa-skull-crossbones" style="color: var(--error-color)"></i>';
            break;
        default:
            levelIcon = '<i class="fas fa-info-circle"></i>';
    }
    
    modalBody.innerHTML = `
        <div class="log-detail">
            <div class="detail-row">
                <div class="detail-label">时间</div>
                <div class="detail-value">${timestamp}</div>
            </div>
            
            <div class="detail-row">
                <div class="detail-label">级别</div>
                <div class="detail-value">
                    ${levelIcon} <span class="log-level ${log.level}">${log.level}</span>
                </div>
            </div>
            
            ${log.source ? `
            <div class="detail-row">
                <div class="detail-label">来源</div>
                <div class="detail-value">${log.source}</div>
            </div>
            ` : ''}
            
            <div class="detail-row">
                <div class="detail-label">消息内容</div>
                <div class="detail-value message">${log.message}</div>
            </div>
        </div>
    `;
    
    confirmAction.style.display = 'none';
    cancelAction.textContent = '关闭';
    
    showModal();
}

/**
 * 显示确认对话框
 */
function showConfirmation(title, message, action) {
    modalTitle.textContent = title;
    modalBody.innerHTML = `<p>${message}</p>`;
    
    confirmAction.style.display = 'block';
    confirmAction.textContent = '确认';
    cancelAction.textContent = '取消';
    
    currentAction = action;
    
    showModal();
}

/**
 * 显示模态框
 */
function showModal() {
    actionModal.classList.add('show');
}

/**
 * 隐藏模态框
 */
function hideModal() {
    actionModal.classList.remove('show');
    currentAction = null;
}

/**
 * 执行维护操作
 */
async function executeAction() {
    let statusMessage = '';
    let actionDescriptionForLoading = '执行操作中...';
    let success = false;

    hideModal(); // Hide modal immediately
    showGlobalLoading(actionDescriptionForLoading);

    try {
        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        switch (currentAction) {
            case 'cleanCache':
                actionDescriptionForLoading = '正在清理缓存...';
                statusMessage = '缓存清理完成！共清理了56个临时文件和734MB空间。';
                success = true;
                break;
            case 'dbMaintenance':
                actionDescriptionForLoading = '正在执行数据库维护...';
                statusMessage = '数据库维护完成！数据库结构已优化，查询性能提升约15%。';
                success = true;
                break;
            case 'restartServices':
                actionDescriptionForLoading = '正在重启服务...';
                statusMessage = '系统服务已成功重启！所有服务现在正常运行。';
                success = true;
                break;
            case 'securityScan':
                actionDescriptionForLoading = '正在执行安全扫描...';
                statusMessage = '安全扫描完成！未发现严重安全隐患。2个低风险警告已记录。';
                success = true;
                break;
            default:
                showGlobalError('未知的维护操作。');
                break;
        }

        if (success) {
            showGlobalSuccess(statusMessage);
        } else if (currentAction) {
            showGlobalError(`操作 ${currentAction} 失败。`);
        }
    } catch (err) {
        showGlobalError(`执行操作 ${currentAction} 失败: ${err.message}`);
        console.error(`Error during ${currentAction}:`, err);
    } finally {
        hideGlobalLoading();
        currentAction = null; // Reset current action
    }
}

/**
 * 更新系统状态指标（模拟）
 */
function updateSystemStatus() {
    // 模拟状态更新
    const updateStats = () => {
        const cpu = Math.floor(Math.random() * 40) + 10;
        const memory = Math.floor(Math.random() * 30) + 25;
        const disk = Math.floor(Math.random() * 10) + 35;
        
        cpuUsage.textContent = `${cpu}%`;
        memoryUsage.textContent = `${memory}%`;
        diskUsage.textContent = `${disk}%`;
        
        // 根据CPU使用率设置颜色状态
        if (cpu > 80) {
            cpuUsage.style.color = 'var(--error-color)';
        } else if (cpu > 60) {
            cpuUsage.style.color = 'var(--warning-color)';
        } else {
            cpuUsage.style.color = 'var(--success-color)';
        }
    };
    
    // 初始更新
    updateStats();
    
    // 定时更新
    setInterval(updateStats, 10000);
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);
