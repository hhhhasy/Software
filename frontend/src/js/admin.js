import session from '../utils/session.js';
import { showLoading, hideLoading, showError, showSuccess } from '../utils/uiUtils.js';

// 编辑用户模态框
const editModal = document.getElementById('editUserModal');
const editUserForm = document.getElementById('editUserForm');

/**
 * 加载用户列表
 */
async function loadUsers() {
  showLoading();
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
    showError('加载用户失败：' + err.message);
    console.error('加载用户失败：', err);
  } finally {
    hideLoading();
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
      showLoading();
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
        showError('加载用户信息失败：' + err.message);
        console.error(err);
      } finally {
        hideLoading();
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
          showError('不能删除自己！');
          return;
      }
      showLoading();
      try {
        const delRes = await fetch(`http://localhost:8000/api/users/${id}`, {
          method: 'DELETE',
          headers: {
            'X-User-ID': currentUser ? currentUser.id.toString() : '0'  // 添加当前用户ID到请求头
          }
        });
        
        if (!delRes.ok) throw new Error('删除失败: ' + delRes.status);
        showSuccess('用户删除成功！');
        // 删除成功后，重新加载列表
        loadUsers();
      } catch (err) {
        showError('删除出错：' + err.message);
        console.error(err);
      } finally {
        hideLoading();
      }
    });
  });
}


// 提交编辑表单
editUserForm.addEventListener('submit', async e => {
  e.preventDefault();
  showLoading();
  
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
    showSuccess('用户信息更新成功！');
  } catch (err) {
    showError('更新用户信息失败：' + err.message);
    console.error(err);
  } finally {
    hideLoading();
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