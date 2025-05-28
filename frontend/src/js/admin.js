import session from '../utils/session.js';
import { showLoading as showGlobalLoading, hideLoading as hideGlobalLoading, showError as showGlobalError, showSuccess as showGlobalSuccess } from '../utils/uiUtils.js';

// 配置
const API_URL = 'http://localhost:8000/api';
const LOGS_PER_PAGE = 50;

// 日志状态变量
let currentLogPage = 1;
let totalLogPages = 1;
let currentLogLevel = '';

// 编辑用户模态框
const editModal = document.getElementById('editUserModal');
const editUserForm = document.getElementById('editUserForm');

// 日志相关DOM元素
const logContainer = document.getElementById('logContainer');
const logLevelFilter = document.getElementById('logLevelFilter');
const refreshLogsBtn = document.getElementById('refreshLogs');
const prevLogsBtn = document.getElementById('prevLogs');
const nextLogsBtn = document.getElementById('nextLogs');
const pageInfo = document.getElementById('pageInfo');

/**
 * 加载用户列表
 */
async function loadUsers() {
  showGlobalLoading();
  try {
    // 用 POST 请求获取用户列表
    const res = await fetch('http://localhost:8000/api/users', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!res.ok) throw new Error('网络错误: ' + res.status);
    const users = await res.json();

    // 渲染到表格
    const tbody = document.getElementById('userList');
    if (!tbody) return;
    
    tbody.innerHTML = '';  // 清空当前内容

    users.forEach(user => {
      const tr = document.createElement('tr');

      // 用户名列
      const tdName = document.createElement('td');
      tdName.textContent = user.username;
      tr.appendChild(tdName);

      // 角色列
      const tdRole = document.createElement('td');
      const span = document.createElement('span');
      span.classList.add('role-tag');
      span.textContent = user.role;
      tdRole.appendChild(span);
      tr.appendChild(tdRole);

      // 操作列（可通过 data-id 获取 user.id 进行编辑/删除）
      const tdAct = document.createElement('td');
      tdAct.innerHTML = `
        <div class="action-buttons" data-id="${user.id}">
          <button class="btn-edit">✏️ 编辑</button>
          <button class="btn-delete">🗑️ 删除</button>
        </div>`;
      tr.appendChild(tdAct);

      tbody.appendChild(tr);
    });

    // 绑定编辑/删除按钮事件
    bindActionButtons();
  } catch (err) {
    showGlobalError('加载用户失败：' + err.message);
    console.error('加载用户失败：', err);
  } finally {
    hideGlobalLoading();
  }
}

/**
 * 绑定用户操作按钮事件
 */
function bindActionButtons() {
  // 编辑按钮
  document.querySelectorAll('.btn-edit').forEach(btn => {
    btn.addEventListener('click', async e => {
      const id = e.currentTarget.closest('.action-buttons').dataset.id;
      showGlobalLoading();
      try {
        // 获取用户详情
        const res = await fetch(`http://localhost:8000/api/users/${id}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        });
        
        if (!res.ok) throw new Error('获取用户信息失败');
        const user = await res.json();

        // 填充表单
        document.getElementById('editUserId').value = user.id;
        document.getElementById('editUsername').value = user.username;
        document.getElementById('editPassword').value = user.password;
        document.getElementById('editRole').value = user.role;

        // 显示模态框
        editModal.style.display = 'block';
      } catch (err) {
        showGlobalError('加载用户信息失败：' + err.message);
        console.error(err);
      } finally {
        hideGlobalLoading();
      }
    });
  });
  
  // 删除按钮
  document.querySelectorAll('.btn-delete').forEach(btn => {
    btn.addEventListener('click', async e => {
      const id = e.currentTarget.closest('.action-buttons').dataset.id;
      if (!confirm('确定要删除此用户？')) return;
      const currentUser = session.get('currentUser');
      if (currentUser && currentUser.id == id) {
          showGlobalError('不能删除自己！');
          return;
      }
      showGlobalLoading();
      try {
        const delRes = await fetch(`http://localhost:8000/api/users/${id}`, {
          method: 'DELETE',
          headers: {
            'X-User-ID': currentUser ? currentUser.id.toString() : '0'  // 添加当前用户ID到请求头
          }
        });
        
        if (!delRes.ok) throw new Error('删除失败: ' + delRes.status);
        showGlobalSuccess('用户删除成功！');
        // 删除成功后，重新加载列表
        loadUsers();
      } catch (err) {
        showGlobalError('删除出错：' + err.message);
        console.error(err);
      } finally {
        hideGlobalLoading();
      }
    });
  });
}


// 提交编辑表单
editUserForm.addEventListener('submit', async e => {
  e.preventDefault();
  showGlobalLoading();
  
  const userId = document.getElementById('editUserId').value;
  const username = document.getElementById('editUsername').value;
  const password = document.getElementById('editPassword').value;
  const role = document.getElementById('editRole').value;

  try {
    const res = await fetch(`http://localhost:8000/api/users/${userId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        username,
        password,
        role
      })
    });

    if (!res.ok) throw new Error('更新失败: ' + res.status);
    
    // 关闭模态框并刷新列表
    editModal.style.display = 'none';
    loadUsers();
    showGlobalSuccess('用户信息更新成功！');
  } catch (err) {
    showGlobalError('更新用户信息失败：' + err.message);
    console.error(err);
  } finally {
    hideGlobalLoading();
  }
});

// 关闭模态框
document.querySelector('.close').addEventListener('click', () => {
  editModal.style.display = 'none';
});

// 点击模态框外部关闭
window.addEventListener('click', (e) => {
  if (e.target === editModal) {
    editModal.style.display = 'none';
  }
});

/**
 * 初始化管理员退出登录按钮
 */
function initAdminLogout() {
  const logoutBtn = document.getElementById('logout');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      // 先移除特定的会话数据，而不是清除所有localStorage
      session.remove('currentUser');
      session.remove('lastActivity');
      // 使用replace而不是href，确保完全重新加载页面
      window.location.replace('login.html');
    });
  }
}

/**
 * 显示当前管理员信息
 */
function displayAdminInfo() {
  const adminSpan = document.getElementById('adminUser');
  if (adminSpan) {
    const currentUser = session.get('currentUser');
    if (currentUser) {
      adminSpan.textContent = currentUser.username;
    }
  }
}





/**
 * 加载系统日志
 */
async function loadLogs() {
  try {
    showGlobalLoading('日志加载中...');
    
    // 构建请求参数
    const requestData = {
      limit: LOGS_PER_PAGE,
      page: currentLogPage
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
  // 创建模态框
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.style.display = 'block';
  
  // 根据日志级别设置不同的图标和颜色
  let levelIcon = '';
  switch(log.level) {
    case 'INFO':
      levelIcon = '<i class="fas fa-info-circle" style="color: var(--primary)"></i>';
      break;
    case 'WARNING':
      levelIcon = '<i class="fas fa-exclamation-triangle" style="color: var(--warning, #faad14)"></i>';
      break;
    case 'ERROR':
      levelIcon = '<i class="fas fa-times-circle" style="color: var(--danger)"></i>';
      break;
    case 'DEBUG':
      levelIcon = '<i class="fas fa-bug" style="color: #93979f"></i>';
      break;
    case 'CRITICAL':
      levelIcon = '<i class="fas fa-skull-crossbones" style="color: var(--danger)"></i>';
      break;
    default:
      levelIcon = '<i class="fas fa-info-circle"></i>';
  }
  
  // 设置模态框内容
  modal.innerHTML = `
    <div class="modal-content">
      <span class="close">&times;</span>
      <h3>日志详情</h3>
      <div class="log-detail">
        <div class="detail-row">
          <div class="detail-label">时间</div>
          <div class="detail-value">${log.timestamp}</div>
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
    </div>
  `;
  
  // 添加到页面
  document.body.appendChild(modal);
  
  // 关闭按钮事件
  const closeBtn = modal.querySelector('.close');
  closeBtn.addEventListener('click', () => {
    document.body.removeChild(modal);
  });
  
  // 点击模态框外部关闭
  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      document.body.removeChild(modal);
    }
  });
}

/**
 * 绑定日志相关事件
 */
function bindLogEvents() {
  const logLevelFilter = document.getElementById('logLevelFilter');
  const refreshLogsBtn = document.getElementById('refreshLogs');
  const prevLogsBtn = document.getElementById('prevLogs');
  const nextLogsBtn = document.getElementById('nextLogs');
  
  // 日志级别筛选
  logLevelFilter.addEventListener('change', () => {
    currentLogLevel = logLevelFilter.value;
    currentLogPage = 1;
    loadLogs();
  });
  
  // 刷新日志
  refreshLogsBtn.addEventListener('click', () => {
    loadLogs();
  });
  
  // 上一页
  prevLogsBtn.addEventListener('click', () => {
    if (currentLogPage > 1) {
      currentLogPage--;
      loadLogs();
    }
  });
  
  // 下一页
  nextLogsBtn.addEventListener('click', () => {
    if (currentLogPage < totalLogPages) {
      currentLogPage++;
      loadLogs();
    }
  });
}

/**
 * 初始化管理员页面
 */
function initAdminPage() {
  displayAdminInfo();
  initAdminLogout();
  loadUsers();
  bindLogEvents();
  loadLogs(); // 加载日志
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initAdminPage);

// 导出函数供其他模块使用
export { loadUsers, loadLogs };