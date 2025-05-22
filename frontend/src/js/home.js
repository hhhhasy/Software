import session from '../utils/session.js';
import { showLoading, hideLoading, showError, showSuccess } from '../utils/uiUtils.js';

/**
 * 初始化音乐播放器
 */
function initMusicPlayer() {
  const audio    = document.getElementById('audioTrack');
  const playBtn  = document.getElementById('playPauseBtn');
  const stopBtn  = document.getElementById('stopBtn');
  const ppIcon   = document.getElementById('ppIcon');
  const prevBtn  = document.getElementById('prevBtn');
  const nextBtn  = document.getElementById('nextBtn');
  const progressFill = document.querySelector('.music-player-card .progress-fill');
  const currentTime = document.querySelector('.time-current');
  const totalTime = document.querySelector('.time-total');

  if (!audio) return;

  const playlist = [
    "../assets/audio/song1.mp3"
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
      ppIcon.classList.remove('fa-play');
      ppIcon.classList.add('fa-pause');
    } else {
      ppIcon.classList.remove('fa-pause');
      ppIcon.classList.add('fa-play');
    }
    
    // 高亮当前歌曲
    document.querySelectorAll('.song-item').forEach((item, idx) => {
      if (idx === currentIndex) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });
  }
  
  // 更新进度条和时间
  function updateProgress() {
    if (!audio.duration) return;
    
    const percent = (audio.currentTime / audio.duration) * 100;
    progressFill.style.width = `${percent}%`;
    
    // 更新时间显示
    const currentMinutes = Math.floor(audio.currentTime / 60);
    const currentSeconds = Math.floor(audio.currentTime % 60).toString().padStart(2, '0');
    currentTime.textContent = `${currentMinutes}:${currentSeconds}`;
    
    const totalMinutes = Math.floor(audio.duration / 60);
    const totalSeconds = Math.floor(audio.duration % 60).toString().padStart(2, '0');
    totalTime.textContent = `${totalMinutes}:${totalSeconds}`;
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

  // 点击专辑图标
  const albumCover = document.querySelector('.album-cover');
  if (albumCover) {
    albumCover.addEventListener('click', () => playBtn.click());
  }

  // 停止按钮
  stopBtn.addEventListener('click', () => {
    audio.pause();
    audio.currentTime = 0;
    updateUI(false);
    updateProgress();
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
  audio.addEventListener('ended', () => {
    nextBtn.click(); // 自动播放下一首
  });
  
  // 更新进度条
  audio.addEventListener('timeupdate', updateProgress);
  
  // 点击播放列表项
  document.querySelectorAll('.song-item').forEach((item, idx) => {
    item.addEventListener('click', () => {
      currentIndex = idx;
      loadTrack(currentIndex);
      audio.play();
      updateUI(true);
    });
  });

  // 初始加载第一首
  loadTrack(currentIndex);
}

/**
 * 初始化标签页切换
 */
function initTabSwitching() {
  const navItems = document.querySelectorAll('.nav-item');
  const panels = document.querySelectorAll('.panel');
  
  if (!navItems.length) return;

  function switchTab(element) {
    const targetPanelId = element.getAttribute('data-panel');
    
    // 移除所有激活状态
    navItems.forEach(item => item.classList.remove('active'));
    panels.forEach(panel => panel.classList.remove('active'));
    
    // 设置当前激活状态
    element.classList.add('active');
    document.getElementById(targetPanelId)?.classList.add('active');
  }

  // 添加点击事件监听器
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      switchTab(item);
    });
  });
  
  // 初始化 - 默认激活第一个标签
  const activeItem = document.querySelector('.nav-item.active') || navItems[0];
  switchTab(activeItem);
}

/**
 * 初始化语音控制功能
 */
function initVoiceControl() {
  const voiceBtn = document.getElementById('voiceBtn');
  const voiceOutput = document.getElementById('textInput');
  const voiceWaves = document.querySelectorAll('.voice-wave');
  
  if (!voiceBtn) return;
  
  let isRecording = false;
  let mediaRecorder;
  let audioChunks = [];
  let currentStream = null;
  
  voiceBtn.addEventListener('click', async () => {
    isRecording = !isRecording;
    
    if (isRecording) {
      showLoading('正在开启语音识别...');
      try {
        currentStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(currentStream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
          audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
          hideLoading();
          showLoading('正在识别语音...');
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
          const formData = new FormData();
          formData.append('audio', audioBlob, 'recording.wav');

          try {
            const currentUser = session.get('currentUser');
            const response = await fetch('http://localhost:8000/api/speech-to-text', {
              method: 'POST',
              headers: {
                'X-User-ID': currentUser?.id
              },
              body: formData
            });

            if (!response.ok) throw await response.json();

            const { command, text } = await response.json();
            voiceOutput.textContent = text;
            showSuccess('语音识别成功: ' + text);
            // 在这里可以根据 command 或 text 执行相应操作
            // 例如: handleDriverVoiceCommand(command, text);
          } catch (err) {
            showError('语音识别失败: ' + (err.detail || '服务器错误'));
            console.error('语音识别错误:', err);
          } finally {
            hideLoading();
          }
        };

        mediaRecorder.start();
        hideLoading();
        voiceBtn.classList.add('recording');
        voiceBtn.innerHTML = '<i class="fas fa-microphone-slash"></i><span>点击停止</span>';
        voiceOutput.textContent = '正在聆听...';
        voiceWaves.forEach((wave, index) => {
          wave.style.height = '20px';
          wave.style.width = '4px';
          wave.style.opacity = '1';
          wave.style.animation = `voiceWave 1s ease-in-out ${index * 0.1}s infinite alternate`;
        });

        // 语音识别超时或自动停止逻辑 (可选)
        // setTimeout(() => {
        //   if (isRecording && mediaRecorder && mediaRecorder.state === "recording") {
        //     mediaRecorder.stop();
        //     if (currentStream) currentStream.getTracks().forEach(track => track.stop());
        //     isRecording = false;
        //     voiceBtn.classList.remove('recording');
        //     voiceBtn.innerHTML = '<i class="fas fa-microphone"></i><span>按下说话</span>';
        //     voiceOutput.textContent = '语音识别已停止';
        //     voiceWaves.forEach(wave => {
        //       wave.style.height = '3px';
        //       wave.style.width = '3px';
        //       wave.style.animation = 'none';
        //     });
        //     showInfo('语音识别超时，已自动停止');
        //   }
        // }, 15000); // 例如15秒超时

      } catch (err) {
        hideLoading();
        showError('无法访问麦克风: ' + err.message);
        console.error('录音错误:', err);
        isRecording = false; // 重置状态
      }
    } else {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
      }
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
      }
      voiceBtn.classList.remove('recording');
      voiceBtn.innerHTML = '<i class="fas fa-microphone"></i><span>按下说话</span>';
      voiceOutput.textContent = '语音识别已停止';
      voiceWaves.forEach(wave => {
        wave.style.height = '3px';
        wave.style.width = '3px';
        wave.style.animation = 'none';
      });
      hideLoading(); // 确保在停止时也隐藏加载指示
    }
  });
  
  // 添加波形动画样式
  const styleElement = document.createElement('style');
  styleElement.innerHTML = `
    @keyframes voiceWave {
      0% { height: 5px; }
      50% { height: 20px; }
      100% { height: 5px; }
    }
  `;
  document.head.appendChild(styleElement);
}

/**
 * 初始化进度环动画
 */
function initSpeedometer() {
  const speedRing = document.getElementById('speedRing');
  const currentSpeed = document.getElementById('currentSpeed');
  
  if (!speedRing || !currentSpeed) return;
  
  // 获取圆环周长
  const radius = speedRing.getAttribute('r');
  const circumference = 2 * Math.PI * radius;
  
  // 设置描边长度等于周长
  speedRing.style.strokeDasharray = `${circumference} ${circumference}`;
  
  // 更新速度表盘
  function updateSpeedometer(speed) {
    // 假设最大速度为200
    const maxSpeed = 200;
    const percent = Math.min(speed / maxSpeed, 1);
    const offset = circumference - percent * circumference;
    
    // 设置偏移量实现填充效果
    speedRing.style.strokeDashoffset = offset;
    currentSpeed.textContent = speed;
    
    // 根据速度变化颜色
    if (speed > 120) {
      speedRing.style.stroke = 'var(--danger)';
    } else if (speed > 80) {
      speedRing.style.stroke = 'var(--warning)';
    } else {
      speedRing.style.stroke = 'var(--primary)';
    }
  }
  
  // 初始化速度值
  updateSpeedometer(65);
  
  // 模拟速度变化
  let count = 0;
  setInterval(() => {
    count++;
    // 模拟速度在40-100之间波动
    const speed = 55 + Math.sin(count / 10) * 15;
    updateSpeedometer(Math.round(speed));
  }, 1000);
}

/**
 * 初始化燃油表
 */
function initFuelGauge() {
  const fuelBar = document.getElementById('fuelBar');
  const fuelLevel = document.getElementById('fuelLevel');
  
  if (!fuelBar || !fuelLevel) return;
  
  // 模拟燃油量逐渐减少
  let fuel = 75;
  setInterval(() => {
    fuel -= 0.1;
    if (fuel < 0) fuel = 75; // 重置
    
    fuelBar.style.width = `${fuel}%`;
    fuelLevel.textContent = Math.round(fuel);
    
    // 根据燃油量变化颜色
    if (fuel < 20) {
      fuelBar.style.background = 'linear-gradient(90deg, var(--danger), #ff7675)';
    } else if (fuel < 40) {
      fuelBar.style.background = 'linear-gradient(90deg, var(--warning), #ffd166)';
    } else {
      fuelBar.style.background = 'linear-gradient(90deg, var(--primary), var(--secondary))';
    }
  }, 1000);
}

/**
 * 初始化系统时间
 */
function initSystemTime() {
  const timeElement = document.getElementById('systemTime');
  
  if (!timeElement) return;
  
  function updateTime() {
    const now = new Date();
    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const seconds = now.getSeconds().toString().padStart(2, '0');
    timeElement.textContent = `${hours}:${minutes}:${seconds}`;
  }
  
  // 立即更新一次
  updateTime();
  
  // 每秒更新一次
  setInterval(updateTime, 1000);
}

/**
 * 初始化天气动画
 */
function initWeatherAnimation() {
  const weatherElements = document.querySelectorAll('.weather-icon');
  
  weatherElements.forEach(icon => {
    // 添加悬浮动画
    icon.style.animation = 'weatherFloat 3s ease-in-out infinite alternate';
  });
  
  // 添加天气动画样式
  const styleElement = document.createElement('style');
  styleElement.innerHTML = `
    @keyframes weatherFloat {
      0% { transform: translateY(0) rotate(0); }
      50% { transform: translateY(-5px) rotate(5deg); }
      100% { transform: translateY(0) rotate(0); }
    }
  `;
  document.head.appendChild(styleElement);
}

/**
 * 初始化退出登录按钮
 */
function initLogout() {
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
 * 添加卡片悬浮效果
 */
function initCardHoverEffects() {
  const cards = document.querySelectorAll('.card');
  
  cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // 根据鼠标位置计算倾斜角度
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = (y - centerY) / 20;
      const rotateY = (centerX - x) / 20;
      
      // 应用3D变换
      card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
    });
    
    card.addEventListener('mouseleave', () => {
      // 重置变换
      card.style.transform = 'translateY(-3px)';
      setTimeout(() => {
        card.style.transform = '';
      }, 300);
    });
  });
}

/**
 * 初始化主界面
 */
function initHomePage() {
  displayUserInfo();
  initLogout();
  initTabSwitching();
  initMusicPlayer();
  initVoiceControl();
  initSpeedometer();
  initFuelGauge();
  initSystemTime();
  initWeatherAnimation();
  
  // 只在桌面端启用卡片悬浮特效
  if (window.innerWidth > 768) {
    initCardHoverEffects();
  }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initHomePage);

// 导出函数供其他模块使用
export { initMusicPlayer, initTabSwitching };