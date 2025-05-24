/**
 * 认证检查模块 - 验证用户是否已登录及访问权限
 */
import session from '../utils/session.js';

/**
 * 页面角色权限配置
 */
const PAGE_ACCESS = {
  'login.html': ['guest'],
  'register.html': ['guest'],
  'driver.html': ['driver'],
  'passenger.html': ['user'],
  'admin.html': ['admin']
};

/**
 * 获取当前页面名称
 */
function getCurrentPage() {
  const path = window.location.pathname;
  const pageName = path.substring(path.lastIndexOf('/') + 1);
  return pageName || 'login.html';
}

/**
 * 检查用户是否有权限访问当前页面
 */
function hasPageAccess(role, page) {
  // 如果页面不在配置中，默认允许访问
  if (!PAGE_ACCESS[page]) return true;
  
  // 未登录用户只能访问guest页面
  if (!role) return PAGE_ACCESS[page].includes('guest');
  
  return PAGE_ACCESS[page].includes(role);
}

/**
 * 重定向到用户对应的首页
 */
function redirectToDriverPage(role) {
  switch (role) {
    case 'admin':
      window.location.href = 'admin.html';
      break;
    case 'driver':
      window.location.href = 'driver.html';
      break;
    case 'user':
      window.location.href = 'passenger.html';
      break;
    default:
      window.location.href = 'login.html';
  }
}

/**
 * 完整的会话和权限检查
 */
function checkAuthentication() {
  const currentPage = getCurrentPage();
  const currentUser = session.get('currentUser');
  const userRole = currentUser ? currentUser.role : 'guest';
  
  console.log(`当前页面: ${currentPage}, 用户角色: ${userRole}`);
  
  // 如果是登录或注册页，且用户已登录，重定向到对应首页
  if ((currentPage === 'login.html' || currentPage === 'register.html') && currentUser) {
    console.log('用户已登录，重定向到首页');
    redirectToDriverPage(userRole);
    return;
  }
  
  // 检查页面访问权限
  if (!hasPageAccess(userRole, currentPage)) {
    console.log('无权限访问当前页面，重定向');
    if (!currentUser) {
      // 未登录用户重定向到登录页
      window.location.href = 'login.html';
    } else {
      // 已登录但权限不符，重定向到对应首页
      redirectToDriverPage(userRole);
    }
    return;
  }
  
  // 登录页面处理过期提示
  if (currentPage === 'login.html') {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('expired') === 'true') {
      // 页面加载后显示会话过期通知
      window.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
          alert('您的会话已过期，请重新登录');
          // 清除URL参数
          window.history.replaceState({}, document.title, window.location.pathname);
        }, 300);
      });
    }
  }
}

// 页面加载时执行认证检查
document.addEventListener('DOMContentLoaded', () => {
  checkAuthentication();
  
  // 添加用户信息显示逻辑
  updateUserDisplay();
});

/**
 * 更新用户信息显示
 */
function updateUserDisplay() {
  const currentUser = session.get('currentUser');
  
  // 更新驾驶员页面用户信息
  const driverUserSpan = document.getElementById('currentUser');
  if (driverUserSpan && currentUser) {
    if (currentUser.role === 'driver') {
      driverUserSpan.textContent = `驾驶员: ${currentUser.username}`;
    } else if (currentUser.role === 'user') {
      driverUserSpan.textContent = `乘客: ${currentUser.username}`;
    }
  }
  
  // 更新管理员页面用户信息
  const adminUserSpan = document.getElementById('adminUser');
  if (adminUserSpan && currentUser && currentUser.role === 'admin') {
    adminUserSpan.textContent = currentUser.username;
  }
}

// 添加定期会话检查
setInterval(() => {
  // 如果会话无效且不在登录/注册页，重定向到登录页
  if (!session.isValid()) {
    const currentPage = getCurrentPage();
    if (currentPage !== 'login.html' && currentPage !== 'register.html') {
      window.location.href = 'login.html?expired=true';
    }
  }
}, 60000); // 每分钟检查一次

export { checkAuthentication, hasPageAccess, updateUserDisplay };