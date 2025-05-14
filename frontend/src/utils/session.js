/**
 * 会话状态管理工具
 */
const session = {
  // 会话超时设置（30分钟）
  SESSION_TIMEOUT: 30 * 60 * 1000,
  
  /**
   * 存储会话数据
   */
  set: (key, value, withTimestamp = true) => {
    // 添加时间戳用于会话超时检查
    const data = withTimestamp ? {
      value,
      timestamp: Date.now()
    } : value;
    
    localStorage.setItem(key, JSON.stringify(data));
  },
  
  /**
   * 获取会话数据
   */
  get: (key, checkExpiry = true) => {
    const dataStr = localStorage.getItem(key);
    if (!dataStr) return null;
    
    try {
      const data = JSON.parse(dataStr);
      
      // 检查是否是带时间戳的数据结构
      if (checkExpiry && data && data.hasOwnProperty('timestamp')) {
        // 检查会话是否过期
        if (Date.now() - data.timestamp > session.SESSION_TIMEOUT) {
          console.log('会话已过期');
          localStorage.removeItem(key);
          return null;
        }
        // 更新时间戳
        session.refreshTimestamp(key);
        return data.value;
      }
      
      return data;
    } catch (err) {
      console.error('会话数据解析错误:', err);
      return null;
    }
  },
  
  /**
   * 刷新会话时间戳
   */
  refreshTimestamp: (key) => {
    const dataStr = localStorage.getItem(key);
    if (!dataStr) return;
    
    try {
      const data = JSON.parse(dataStr);
      if (data && data.hasOwnProperty('timestamp') && data.hasOwnProperty('value')) {
        data.timestamp = Date.now();
        localStorage.setItem(key, JSON.stringify(data));
      }
    } catch (err) {
      console.error('刷新会话时间戳错误:', err);
    }
  },
  
  /**
   * 检查会话是否有效
   */
  isValid: () => {
    const currentUser = session.get('currentUser', true);
    return !!currentUser;
  },
  
  /**
   * 获取当前用户角色
   */
  getUserRole: () => {
    const currentUser = session.get('currentUser');
    return currentUser ? currentUser.role : null;
  },
  
  /**
   * 清除所有会话数据
   */
  clear: () => localStorage.clear(),
  
  /**
   * 删除特定会话数据
   */
  remove: (key) => localStorage.removeItem(key)
};

// 添加会话活动监听
if (typeof window !== 'undefined') {
  // 用户活动时刷新会话
  ['click', 'keypress', 'scroll', 'mousemove'].forEach(evt => {
    window.addEventListener(evt, () => {
      if (session.isValid()) {
        session.refreshTimestamp('currentUser');
      }
    });
  });
  
  // 定期检查会话是否过期（每分钟）
  setInterval(() => {
    if (!session.isValid()) {
      // 如果会话过期且不在登录页，则重定向
      if (!window.location.pathname.includes('login.html') && 
          !window.location.pathname.includes('register.html')) {
        window.location.href = 'login.html?expired=true';
      }
    }
  }, 60000);
}

// 导出会话管理工具
export default session;