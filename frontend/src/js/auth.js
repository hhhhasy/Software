import session from '../utils/session.js';

// 处理登录表单提交
async function handleLogin(e) {
  e.preventDefault();
  
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;

  try {
    const response = await fetch('http://localhost:8000/api/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password})
    });

    if (!response.ok) throw await response.json();
    
    const { role } = await response.json();
    session.set('currentUser', { username, role });
    
    if(role=='admin'){
      window.location.href='admin.html'
    }

    if(role=='driver'){
      window.location.href='home.html'
    }

    if(role=='user'){
      window.location.href='passenger.html'
    }
  } catch (err) {
    alert(err.detail || '登录失败');
  }
}

// 处理注册表单提交
async function handleRegister(e) {
  e.preventDefault();
  
  const newUser = {
    username: document.getElementById('newUsername').value,
    password: document.getElementById('newPassword').value,
    confirm_password: document.getElementById('confirmPassword').value
  };

  try {
    const response = await fetch('http://localhost:8000/api/register', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(newUser)
    });

    if (!response.ok) throw await response.json();
    
    alert('注册成功，请登录');
    window.location.href = 'login.html';
  } catch (err) {
    alert(err.detail || '注册失败');
  }
}

// 初始化认证页面
function initAuth() {
  // 监听登录表单
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', handleLogin);
  }
  
  // 监听注册表单
  const registerForm = document.getElementById('registerForm');
  if (registerForm) {
    registerForm.addEventListener('submit', handleRegister);
  }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initAuth);

// 导出函数供其他模块使用
export { handleLogin, handleRegister };