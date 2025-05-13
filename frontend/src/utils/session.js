/**
 * 会话状态管理工具
 */
const session = {

  set: (key, value) => localStorage.setItem(key, JSON.stringify(value)),
  
  get: (key) => {
    const data = localStorage.getItem(key);
    return data ? JSON.parse(data) : null;
  },
  
  clear: () => localStorage.clear()
};

// 导出会话管理工具
export default session;