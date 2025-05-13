/**
 * 认证检查模块 - 验证用户是否已登录，否则重定向到登录页
 */
import session from '../utils/session.js';

/**
 * 检查用户是否已登录，未登录则重定向到登录页
 */
function checkAuthentication() {
  const currentUser = session.get('currentUser');
  
  // 如果用户未登录，重定向到登录页
  if (!currentUser) {
    console.log('未检测到登录信息，重定向到登录页');
    window.location.href = 'login.html';
    return;
  }
  
  // 如果在管理员页但不是管理员角色，重定向到普通用户页面
  const isAdminPage = window.location.pathname.includes('admin.html');
  if (isAdminPage && currentUser.role !== 'admin') {
    console.log('无管理员权限，重定向到用户主页');
    window.location.href = 'home.html';
    return;
  }
}

// 页面加载时执行认证检查
document.addEventListener('DOMContentLoaded', checkAuthentication);

export { checkAuthentication };