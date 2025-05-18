import session from '../utils/session.js';

/**
 * 处理登录表单提交
 */
async function handleLogin(e) {
  e.preventDefault();
  
  const loginBtn = document.querySelector('#loginForm button[type="submit"]');
  if (loginBtn) {
    loginBtn.disabled = true;
    loginBtn.innerHTML = '<span class="spinner"></span> 登录中...';
  }
  
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;

  try {
    const response = await fetch('http://localhost:8000/api/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password})
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw errorData;
    }
    
    const {id, role} = await response.json();
    
    // 使用新的会话存储方式，添加时间戳
    session.set('currentUser', { id,username, role })
    session.set('lastActivity', Date.now());
    
    // 基于角色重定向
    if(role === 'admin'){
      window.location.href = 'admin.html';
    } else if(role === 'driver'){
      window.location.href = 'home.html';
    } else if(role === 'user'){
      window.location.href = 'passenger.html';
    } else if(role === 'maintenance_personne'){
      window.location.href = 'maintainer.html';
    } else {
      // 默认重定向到login
      session.clear();
      throw { detail: '无效的用户角色' };
    }
  } catch (err) {
    // 重置按钮状态
    if (loginBtn) {
      loginBtn.disabled = false;
      loginBtn.textContent = '登录';
    }
    
    showAuthError(err.detail || '登录失败，请检查用户名和密码');
  }
}

/**
 * 处理注册表单提交
 */
async function handleRegister(e) {
  e.preventDefault();
  
  const registerBtn = document.querySelector('#registerForm button[type="submit"]');
  if (registerBtn) {
    registerBtn.disabled = true;
    registerBtn.innerHTML = '<span class="spinner"></span> 注册中...';
  }
  
  const newUser = {
    username: document.getElementById('newUsername').value,
    password: document.getElementById('newPassword').value,
    confirm_password: document.getElementById('confirmPassword').value
  };

  try {
    // 前端验证
    if (newUser.password !== newUser.confirm_password) {
      throw { detail: '两次输入的密码不匹配' };
    }
    
    if (newUser.password.length < 6) {
      throw { detail: '密码长度不能少于6位' };
    }

    const response = await fetch('http://localhost:8000/api/register', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(newUser)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw errorData;
    }
    
    showAuthSuccess('注册成功，即将跳转到登录页');
    window.location.href = 'login.html';
  } catch (err) {
    // 重置按钮状态
    if (registerBtn) {
      registerBtn.disabled = false;
      registerBtn.textContent = '立即注册';
    }
    
    showAuthError(err.detail || '注册失败');
  }
}

/**
 * 显示认证错误消息
 */
function showAuthError(message) {
  // 检查是否已存在错误消息元素
  let errorEl = document.querySelector('.auth-error');
  
  if (!errorEl) {
    // 创建错误消息元素
    errorEl = document.createElement('div');
    errorEl.className = 'auth-error';
    
    // 插入到表单前
    const form = document.querySelector('form');
    if (form) {
      form.parentNode.insertBefore(errorEl, form);
    }
  }
  
  errorEl.textContent = message;
  errorEl.style.opacity = '1';
  
  // 5秒后淡出
  setTimeout(() => {
    errorEl.style.opacity = '0';
  }, 5000);
}

/**
 * 显示认证成功消息
 */
function showAuthSuccess(message) {
  // 检查是否已存在成功消息元素
  let successEl = document.querySelector('.auth-success');
  
  if (!successEl) {
    // 创建成功消息元素
    successEl = document.createElement('div');
    successEl.className = 'auth-success';
    
    // 插入到表单前
    const form = document.querySelector('form');
    if (form) {
      form.parentNode.insertBefore(successEl, form);
    }
  }
  
  successEl.textContent = message;
  successEl.style.opacity = '1';
}

/**
 * 初始化认证页面
 */
function initAuth() {
  // 添加登录表单提交事件
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', handleLogin);
    
    // 检查URL参数是否有error
    const urlParams = new URLSearchParams(window.location.search);
    const errorMsg = urlParams.get('error');
    if (errorMsg) {
      showAuthError(decodeURIComponent(errorMsg));
      // 清除URL参数
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  }
  
  // 添加注册表单提交事件
  const registerForm = document.getElementById('registerForm');
  if (registerForm) {
    registerForm.addEventListener('submit', handleRegister);
  }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initAuth);

// 导出函数供其他模块使用
export { handleLogin, handleRegister };