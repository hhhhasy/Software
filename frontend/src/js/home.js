import session from '../utils/session.js';

/**
 * 初始化音乐播放器
 */
function initMusicPlayer() {
  const audio    = document.getElementById('audioTrack');
  const icon     = document.getElementById('musicIcon');
  const playBtn  = document.getElementById('playPauseBtn');
  const stopBtn  = document.getElementById('stopBtn');
  const ppIcon   = document.getElementById('ppIcon');
  const ppText   = document.getElementById('ppText');
  const prevBtn  = document.getElementById('prevBtn');
  const nextBtn  = document.getElementById('nextBtn');

  if (!audio) return;

  const playlist = [
    "../../backend/audio/song.mp3",
    "../../backend/audio/song1.mp3"
  ];
  let currentIndex = 0;

  // 加载指定索引的音轨
  function loadTrack(index) {
    if (index < 0) index = playlist.length - 1;
    if (index >= playlist.length) index = 0;
    currentIndex = index;

    audio.src = playlist[currentIndex];
    audio.currentTime = 0;
    updateUI(false);
  }

  function updateUI(isPlaying) {
    if (isPlaying) {
      icon.classList.add('rotating');
      ppIcon.textContent = '⏸️';
      ppText.textContent = '暂停';
    } else {
      icon.classList.remove('rotating');
      ppIcon.textContent = '▶️';
      ppText.textContent = '播放';
    }
  }

  // 播放/暂停按钮
  playBtn.addEventListener('click', () => {
    if (audio.paused) {
      audio.play();
      updateUI(true);
    } else {
      audio.pause();
      updateUI(false);
    }
  });

  // 点击音符图标
  icon.addEventListener('click', () => playBtn.click());

  // 停止按钮
  stopBtn.addEventListener('click', () => {
    audio.pause();
    audio.currentTime = 0;
    updateUI(false);
  });

  // 上一首/下一首
  prevBtn.addEventListener('click', () => {
    loadTrack(currentIndex - 1);
    audio.play();
    updateUI(true);
  });

  nextBtn.addEventListener('click', () => {
    loadTrack(currentIndex + 1);
    audio.play();
    updateUI(true);
  });

  // 音乐结束事件
  audio.addEventListener('ended', () => updateUI(false));

  // 初始加载第一首
  loadTrack(currentIndex);
}

/**
 * 初始化标签页切换
 */
function initTabSwitching() {
  const navBtns = document.querySelectorAll('.sidebar button');
  const panels = document.querySelectorAll('.panel');
  
  if (!navBtns.length) return;

  // 定义标签切换处理函数
  function switchTab(buttonId) {
    // 处理按钮激活状态
    navBtns.forEach(b => b.classList.remove('active'));
    document.getElementById(buttonId)?.classList.add('active');
    
    // 隐藏所有面板
    panels.forEach(p => p.classList.add('hidden'));
    
    // 显示对应面板
    switch (buttonId) {
      case 'navDriving':
        document.getElementById('panelDrivingStatus')?.classList.remove('hidden');
        break;
      case 'navMultimodal':
        document.getElementById('panelMultimodal')?.classList.remove('hidden');
        break;
      case 'navMusic':
        document.getElementById('panelMusic')?.classList.remove('hidden');
        break;
    }
  }

  // 添加按钮点击事件监听器
  navBtns.forEach(btn => btn.addEventListener('click', () => {
    switchTab(btn.id);
  }));
  
  // 初始化 - 确保页面加载时显示正确的面板
  // 查找当前激活的标签，或默认使用第一个标签
  const activeBtn = document.querySelector('.sidebar button.active') || navBtns[0];
  switchTab(activeBtn.id);
}

/**
 * 初始化退出登录按钮
 */
function initLogout() {
  const logoutBtn = document.getElementById('logout');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      session.clear();
      window.location.href = 'login.html';
    });
  }
}

/**
 * 显示当前用户信息
 */
function displayUserInfo() {
  const userSpan = document.getElementById('currentUser');
  if (userSpan) {
    const currentUser = session.get('currentUser');
    if (currentUser) {
      userSpan.textContent = `驾驶员: ${currentUser.username}`;
    }
  }
}

/**
 * 初始化主界面
 */
function initHomePage() {
  displayUserInfo();
  initLogout();
  initTabSwitching();
  initMusicPlayer();
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
  displayUserInfo();
  initLogout();
  initTabSwitching(); // 确保这个函数最先执行以设置初始面板显示
  initMusicPlayer();
});

// 导出函数供其他模块使用
export { initMusicPlayer, initTabSwitching };