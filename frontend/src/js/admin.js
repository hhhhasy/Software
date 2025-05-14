import session from '../utils/session.js';

/**
 * 加载用户列表
 */
async function loadUsers() {
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
    console.error('加载用户失败：', err);
  }
}

/**
 * 绑定用户操作按钮事件
 */
function bindActionButtons() {
  // 编辑按钮
  document.querySelectorAll('.btn-edit').forEach(btn => {
    btn.addEventListener('click', e => {
      const id = e.currentTarget.closest('.action-buttons').dataset.id;
      alert(`编辑用户ID: ${id}（功能待实现）`);
    });
  });
  
  // 删除按钮
  document.querySelectorAll('.btn-delete').forEach(btn => {
    btn.addEventListener('click', async e => {
      const id = e.currentTarget.closest('.action-buttons').dataset.id;
      if (!confirm('确定要删除此用户？')) return;
      
      try {
        const delRes = await fetch(`http://localhost:8000/api/users/${id}`, {
          method: 'DELETE'
        });
        
        if (!delRes.ok) throw new Error('删除失败');
        // 删除成功后，重新加载列表
        loadUsers();
      } catch (err) {
        console.error(err);
        alert('删除出错：' + err.message);
      }
    });
  });
}

/**
 * 初始化管理员退出登录按钮
 */
function initAdminLogout() {
  const logoutBtn = document.getElementById('adminLogout');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      session.clear();
      window.location.href = 'login.html';
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
 * 初始化管理员页面
 */
function initAdminPage() {
  displayAdminInfo();
  initAdminLogout();
  loadUsers();
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initAdminPage);

// 导出函数供其他模块使用
export { loadUsers };