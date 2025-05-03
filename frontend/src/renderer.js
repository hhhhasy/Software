// 会话状态管理
const session = {
  set: (key, value) => localStorage.setItem(key, JSON.stringify(value)),
  get: (key) => JSON.parse(localStorage.getItem(key)),
  clear: () => localStorage.clear()
}

// 登录表单处理
document.getElementById('loginForm')?.addEventListener('submit', async (e) => {
  e.preventDefault()
  
  const username = document.getElementById('username').value
  const password = document.getElementById('password').value

  try {
    const response = await fetch('http://localhost:8000/api/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password})
    })

    if (!response.ok) throw await response.json()
    
    const { role } = await response.json()
    session.set('currentUser', { username, role })
    
    window.location.href = role === 'admin' ? 'admin.html' : 'home.html'
  } catch (err) {
    alert(err.detail || '登录失败')
  }
})

// 注册表单处理
document.getElementById('registerForm')?.addEventListener('submit', async (e) => {
  e.preventDefault()
  
  const newUser = {
    username: document.getElementById('newUsername').value,
    password: document.getElementById('newPassword').value,
    confirm_password: document.getElementById('confirmPassword').value
  }

  try {
    const response = await fetch('http://localhost:8000/api/register', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(newUser)
    })

    if (!response.ok) throw await response.json()
    
    alert('注册成功，请登录')
    window.location.href = '../views/login.html'
  } catch (err) {
    alert(err.detail || '注册失败')
  }
})