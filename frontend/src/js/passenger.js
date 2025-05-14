import session from '../utils/session.js';

// 音乐播放控制逻辑
const audio = document.getElementById('audioTrack');
const playBtn = document.getElementById('playPauseBtn');
const ppIcon = document.getElementById('ppIcon');

playBtn.addEventListener('click', () => {
    if (audio.paused) {
        audio.play();
        ppIcon.textContent = '⏸️';
    } else {
        audio.pause();
        ppIcon.textContent = '▶️';
    }
});

/**
 * 初始化退出登录按钮
 */
function initLogout() {
    const logoutBtn = document.getElementById('logout');
    if (logoutBtn) {
      logoutBtn.addEventListener('click', () => {
        window.location.href = 'login.html';
      });
    }
  }


/**
 * 初始化主界面
 */
function initHomePage() {
  initLogout();
}



// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initHomePage);
