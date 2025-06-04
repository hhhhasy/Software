import session from '../utils/session.js';
import { showLoading, hideLoading, showError, showSuccess } from '../utils/uiUtils.js';


// 新增：获取闪光灯相关的DOM元素
const flashOverlayContainer = document.getElementById('flash-overlay-container');
let flashIntervalId = null;
let isFlashing = false;

/**
 * 启动屏幕边缘闪光效果
 * @param {number} duration - 闪光效果持续的总时间 (毫秒)。如果为0或负数，则一直闪烁直到手动停止。
 * @param {number} interval - 一个完整闪烁周期的时长 (毫秒)，例如 500ms 表示亮0.25s灭0.25s。
 * @param {string} color - 闪光的颜色 (可选, 默认 'red')。
 */
function startScreenFlash(duration = 5000, interval = 500, color = 'red') {
  if (isFlashing || !flashOverlayContainer) return; // 如果已经在闪烁或元素不存在，则不重复启动

  console.log(`Starting screen flash. Duration: ${duration}ms, Interval: ${interval}ms, Color: ${color}`);
  isFlashing = true;

  // 设置闪光颜色
  const edges = flashOverlayContainer.querySelectorAll('.flash-edge');
  edges.forEach(edge => edge.style.backgroundColor = color);

  flashOverlayContainer.style.display = 'block'; // 显示闪光灯容器

  let flashStateOn = false; // 控制当前是亮还是灭
  // interval 是一个完整周期（亮+灭），所以切换状态的间隔是 interval / 2
  flashIntervalId = setInterval(() => {
    flashStateOn = !flashStateOn;
    if (flashStateOn) {
      flashOverlayContainer.classList.add('active');
    } else {
      flashOverlayContainer.classList.remove('active');
    }
  }, interval / 2);

  // 如果设置了持续时间，则在时间到后停止
  if (duration > 0) {
    setTimeout(() => {
      stopScreenFlash();
    }, duration);
  }
}

/**
 * 停止屏幕边缘闪光效果
 */
function stopScreenFlash() {
  if (!isFlashing || !flashOverlayContainer) return;
  console.log("Stopping screen flash.");

  if (flashIntervalId) {
    clearInterval(flashIntervalId);
    flashIntervalId = null;
  }
  flashOverlayContainer.classList.remove('active'); // 确保最后是熄灭状态
  flashOverlayContainer.style.display = 'none';    // 隐藏容器
  isFlashing = false;
}

  // 挂载到 window 对象，以便 multimedia.js 可以调用
window.startAppScreenFlash = startScreenFlash;
window.stopAppScreenFlash = stopScreenFlash;

/**
 * 初始化主题切换功能
 */
function initThemeSwitcher() {
  const themeSelector = document.getElementById('themeSelector');
  const bodyElement = document.body;
  const THEME_STORAGE_KEY = 'neurodrive_theme'; // 本地存储的键名

  // 应用已保存的主题或默认主题
  function applyTheme(theme) {
    bodyElement.classList.remove('theme-light', 'theme-dark', 'theme-auto'); // 清除旧的主题类
    switch (theme) {
      case 'light':
        bodyElement.classList.add('theme-light');
        break;
      case 'dark':
        bodyElement.classList.add('theme-dark'); // 可以明确添加一个 .theme-dark，即使它是默认的
        break;
      case 'auto':
        bodyElement.classList.add('theme-auto');
        // 自动模式逻辑 (可以基于系统偏好或时间)
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
          bodyElement.classList.add('theme-light');
        } else {
          bodyElement.classList.add('theme-dark'); // 默认自动为暗色或根据系统
        }
        break;
      default:
        bodyElement.classList.add('theme-dark'); // 默认暗色
    }
    // 更新选择器的显示值
    if (themeSelector) themeSelector.value = theme;
    console.log(`Theme applied: ${theme}`);
  }

  // 加载保存的主题
  const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) || 'dark'; // 默认暗色
  applyTheme(savedTheme);

  if (themeSelector) {
    themeSelector.addEventListener('change', (event) => {
      const selectedTheme = event.target.value;
      applyTheme(selectedTheme);
      localStorage.setItem(THEME_STORAGE_KEY, selectedTheme); // 保存用户选择
      showSuccess(`主题已切换为: ${themeSelector.options[themeSelector.selectedIndex].text}`);
    });
  }

  // (可选) 监听系统颜色方案变化，用于 'auto' 模式
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', e => {
      if (localStorage.getItem(THEME_STORAGE_KEY) === 'auto') {
        applyTheme('auto'); // 重新应用自动模式逻辑
      }
    });
  }
}


/**
 * 初始化亮度调节功能
 */
function initBrightnessControl() {
  const brightnessSlider = document.getElementById('brightnessSlider');
  const brightnessValueDisplay = document.getElementById('brightnessValue');
  const brightnessOverlay = document.getElementById('brightness-overlay');
  const BRIGHTNESS_STORAGE_KEY = 'neurodrive_brightness';

  if (!brightnessSlider || !brightnessOverlay || !brightnessValueDisplay) return;

  // 应用亮度
  function applyBrightness(level) { // level 从 0 到 100
    // level 100 = 最亮 (覆盖层透明 opacity 0)
    // level 0 = 最暗 (覆盖层不透明 opacity 1, 颜色为黑色)
    const overlayOpacity = 1 - (level / 100);
    brightnessOverlay.style.backgroundColor = `rgba(0, 0, 0, ${overlayOpacity})`;
    
    // 更新滑块和显示值
    brightnessSlider.value = level;
    brightnessValueDisplay.textContent = `${level}%`;
    console.log(`Brightness applied: ${level}% (Overlay opacity: ${overlayOpacity})`);
  }

  // 加载保存的亮度
  const savedBrightness = localStorage.getItem(BRIGHTNESS_STORAGE_KEY);
  // 如果有保存的值，则使用；否则使用滑块的默认值 (100)
  const initialBrightness = savedBrightness !== null ? parseInt(savedBrightness, 10) : parseInt(brightnessSlider.value, 10);
  applyBrightness(initialBrightness);

  brightnessSlider.addEventListener('input', (event) => {
    const newBrightness = parseInt(event.target.value, 10);
    applyBrightness(newBrightness);
  });

  brightnessSlider.addEventListener('change', (event) => { // 'change' 事件在用户释放滑块后触发
    const newBrightness = parseInt(event.target.value, 10);
    localStorage.setItem(BRIGHTNESS_STORAGE_KEY, newBrightness.toString());
    showSuccess(`亮度已调整为: ${newBrightness}%`);
  });
}

/**
 * 初始化音量控制功能
 */
function initVolumeControls() {
  const mediaVolumeSlider = document.getElementById('mediaVolumeSlider');
  const mediaVolumeValue = document.getElementById('mediaVolumeValue');
  const mainAudioPlayer = document.getElementById('audioTrack'); // 已有的音乐播放器 <audio>

  const navVolumeSlider = document.getElementById('navVolumeSlider');
  const navVolumeValue = document.getElementById('navVolumeValue');

  const systemSoundsToggle = document.getElementById('systemSoundsToggle');

  const VOLUME_STORAGE_PREFIX = 'neurodrive_volume_';

  // 设置和应用媒体音量
  function applyMediaVolume(level) { // level 0-100
    const volumeFloat = level / 100; // <audio> 元素的 volume 是 0.0 - 1.0
    if (mainAudioPlayer) {
      mainAudioPlayer.volume = volumeFloat;
    }
    mediaVolumeSlider.value = level;
    mediaVolumeValue.textContent = `${level}%`;
    console.log(`Media volume set to: ${volumeFloat}`);
  }

  // 设置和应用导航音量 (概念性 - 需要实际的导航声音控制机制)
  function applyNavVolume(level) {
    const volumeFloat = level / 100;
    navVolumeSlider.value = level;
    navVolumeValue.textContent = `${level}%`;
    console.log(`Navigation volume (conceptual) set to: ${volumeFloat}`);
    // TODO: 调用实际的导航音量控制API或方法
  }

  // 设置和应用系统提示音开关 (概念性)
  function applySystemSoundsToggle(isOn) {
    systemSoundsToggle.checked = isOn;
    console.log(`System sounds toggled: ${isOn ? 'ON' : 'OFF'}`);
    // TODO: 设置一个全局变量或调用方法来实际控制提示音的播放
    // window.playSystemSounds = isOn;
  }


  // --- 媒体音量 ---
  const savedMediaVolume = localStorage.getItem(VOLUME_STORAGE_PREFIX + 'media');
  const initialMediaVolume = savedMediaVolume !== null ? parseInt(savedMediaVolume, 10) : parseInt(mediaVolumeSlider.value, 10);
  applyMediaVolume(initialMediaVolume);

  mediaVolumeSlider.addEventListener('input', (event) => {
    applyMediaVolume(parseInt(event.target.value, 10));
  });
  mediaVolumeSlider.addEventListener('change', (event) => {
    localStorage.setItem(VOLUME_STORAGE_PREFIX + 'media', event.target.value);
    showSuccess(`媒体音量已调整为: ${event.target.value}%`);
  });

  // --- 导航音量 (概念性 需要根据实际调整) ---
  const savedNavVolume = localStorage.getItem(VOLUME_STORAGE_PREFIX + 'navigation');
  const initialNavVolume = savedNavVolume !== null ? parseInt(savedNavVolume, 10) : parseInt(navVolumeSlider.value, 10);
  applyNavVolume(initialNavVolume);

  navVolumeSlider.addEventListener('input', (event) => {
    applyNavVolume(parseInt(event.target.value, 10));
  });
  navVolumeSlider.addEventListener('change', (event) => {
    localStorage.setItem(VOLUME_STORAGE_PREFIX + 'navigation', event.target.value);
    showSuccess(`导航音量已调整为: ${event.target.value}%`);
    // 如果导航音量由后端控制，这里可能需要发送API请求
  });

  // --- 系统提示音开关 (概念性) ---
  const savedSystemSounds = localStorage.getItem(VOLUME_STORAGE_PREFIX + 'system_sounds');
  // 默认为 true (checked)
  const initialSystemSoundsOn = savedSystemSounds !== null ? JSON.parse(savedSystemSounds) : systemSoundsToggle.checked;
  applySystemSoundsToggle(initialSystemSoundsOn);
  window.playSystemSounds = initialSystemSoundsOn; // 设置一个全局变量供其他地方使用

  systemSoundsToggle.addEventListener('change', (event) => {
    const isOn = event.target.checked;
    applySystemSoundsToggle(isOn);
    localStorage.setItem(VOLUME_STORAGE_PREFIX + 'system_sounds', JSON.stringify(isOn));
    window.playSystemSounds = isOn;
    showSuccess(`系统提示音已${isOn ? '开启' : '关闭'}`);
  });
}

/**
 * 初始化智能交互区域的事件监听 (文本输入和常用指令)
 */
function initSmartInteractionInputEvents() {
  const commandTextInput = document.getElementById('commandTextInput');
  const sendCommandBtn = document.getElementById('sendCommandBtn');
  const commonCommandChipsContainer = document.getElementById('commonCommandChips');

  if (!commandTextInput || !sendCommandBtn || !commonCommandChipsContainer) {
    console.warn("智能交互输入区域的必要元素未找到!");
    return;
  }

  // 事件：点击发送按钮
  sendCommandBtn.addEventListener('click', () => {
    if (window.processDriverTextInput) { // 检查 multimedia.js 中的函数是否存在
      window.processDriverTextInput(commandTextInput.value);
    } else {
      showError("文本指令处理功能未初始化。");
    }
  });

  // 事件：在文本输入框中按 Enter 键
  commandTextInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      if (window.processDriverTextInput) {
        window.processDriverTextInput(commandTextInput.value);
      } else {
        showError("文本指令处理功能未初始化。");
      }
    }
  });

  // 事件：点击常用指令
  commonCommandChipsContainer.addEventListener('click', (event) => {
    const targetChip = event.target.closest('.command-chip');
    if (targetChip) {
      const command = targetChip.dataset.command;
      if (command) {
        commandTextInput.value = command;
        commandTextInput.focus();
        showSuccess(`指令 "${command}" 已填充`);
      }
    }
  });

  // 注意：语音按钮(#voiceBtn)的事件监听仍然在 multimedia.js 的 initMultimedia 中处理
}

/**
 * 初始化音乐播放器
 */
function initMusicPlayer() {
  const audio    = document.getElementById('audioTrack');
  const playPauseBtn  = document.getElementById('playPauseBtn');
  const ppIcon   = document.getElementById('ppIcon');
  const prevBtn  = document.getElementById('prevBtn');
  const nextBtn  = document.getElementById('nextBtn');
  const progressFill = document.querySelector('.music-player-card .progress-fill');
  const currentTime = document.querySelector('.time-current');
  const totalTime = document.querySelector('.time-total');

  if (!audio) return;

  const playlist = [
    "../assets/audio/song1.mp3",
    "../assets/audio/song2.mp3",
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
  
  // 更新播放器UI状态
  window.updateDriverMusicPlayerUI = updateUI;

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
  playPauseBtn.addEventListener('click', () => {
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
    albumCover.addEventListener('click', () => playPauseBtn.click());
  }

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
// function initVoiceControl() {
//   const voiceBtn = document.getElementById('voiceBtn');
//   const voiceOutput = document.getElementById('textInput');
//   const voiceWaves = document.querySelectorAll('.voice-wave');
  
//   if (!voiceBtn) return;
  
//   let isRecording = false;
//   let mediaRecorder;
//   let audioChunks = [];
//   let currentStream = null;
  
//   voiceBtn.addEventListener('click', async () => {
//     isRecording = !isRecording;
    
//     if (isRecording) {
//       showLoading('正在开启语音识别...');
//       try {
//         currentStream = await navigator.mediaDevices.getUserMedia({ audio: true });
//         mediaRecorder = new MediaRecorder(currentStream);
//         audioChunks = [];

//         mediaRecorder.ondataavailable = (e) => {
//           audioChunks.push(e.data);
//         };

//         mediaRecorder.onstop = async () => {
//           hideLoading();
//           showLoading('正在识别语音...');
//           const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
//           const formData = new FormData();
//           formData.append('audio', audioBlob, 'recording.wav');

//           try {
//             const currentUser = session.get('currentUser');
//             const response = await fetch('http://localhost:8000/api/speech-to-text', {
//               method: 'POST',
//               headers: {
//                 'X-User-ID': currentUser?.id
//               },
//               body: formData
//             });

//             if (!response.ok) throw await response.json();

//             const { command, text } = await response.json();
//             voiceOutput.textContent = text;
//             showSuccess('语音识别成功: ' + text);
//             // 在这里可以根据 command 或 text 执行相应操作
//             // 例如: handleDriverVoiceCommand(command, text);
//           } catch (err) {
//             showError('语音识别失败: ' + (err.detail || '服务器错误'));
//             console.error('语音识别错误:', err);
//           } finally {
//             hideLoading();
//           }
//         };

//         mediaRecorder.start();
//         hideLoading();
//         voiceBtn.classList.add('recording');
//         voiceBtn.innerHTML = '<i class="fas fa-microphone-slash"></i><span>点击停止</span>';
//         voiceOutput.textContent = '正在聆听...';
//         voiceWaves.forEach((wave, index) => {
//           wave.style.height = '20px';
//           wave.style.width = '4px';
//           wave.style.opacity = '1';
//           wave.style.animation = `voiceWave 1s ease-in-out ${index * 0.1}s infinite alternate`;
//         });

//         // 语音识别超时或自动停止逻辑 (可选)
//         // setTimeout(() => {
//         //   if (isRecording && mediaRecorder && mediaRecorder.state === "recording") {
//         //     mediaRecorder.stop();
//         //     if (currentStream) currentStream.getTracks().forEach(track => track.stop());
//         //     isRecording = false;
//         //     voiceBtn.classList.remove('recording');
//         //     voiceBtn.innerHTML = '<i class="fas fa-microphone"></i><span>按下说话</span>';
//         //     voiceOutput.textContent = '语音识别已停止';
//         //     voiceWaves.forEach(wave => {
//         //       wave.style.height = '3px';
//         //       wave.style.width = '3px';
//         //       wave.style.animation = 'none';
//         //     });
//         //     showInfo('语音识别超时，已自动停止');
//         //   }
//         // }, 15000); // 例如15秒超时

//       } catch (err) {
//         hideLoading();
//         showError('无法访问麦克风: ' + err.message);
//         console.error('录音错误:', err);
//         isRecording = false; // 重置状态
//       }
//     } else {
//       if (mediaRecorder && mediaRecorder.state === "recording") {
//         mediaRecorder.stop();
//       }
//       if (currentStream) {
//         currentStream.getTracks().forEach(track => track.stop());
//       }
//       voiceBtn.classList.remove('recording');
//       voiceBtn.innerHTML = '<i class="fas fa-microphone"></i><span>按下说话</span>';
//       voiceOutput.textContent = '语音识别已停止';
//       voiceWaves.forEach(wave => {
//         wave.style.height = '3px';
//         wave.style.width = '3px';
//         wave.style.animation = 'none';
//       });
//       hideLoading(); // 确保在停止时也隐藏加载指示
//     }
//   });
  
//   // 添加波形动画样式
//   const styleElement = document.createElement('style');
//   styleElement.innerHTML = `
//     @keyframes voiceWave {
//       0% { height: 5px; }
//       50% { height: 20px; }
//       100% { height: 5px; }
//     }
//   `;
//   document.head.appendChild(styleElement);
// }

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
function initDriverPage() {
  displayUserInfo();
  initLogout();
  initTabSwitching();
  initMusicPlayer();
  initSpeedometer();
  initFuelGauge();
  initSystemTime();
  initWeatherAnimation();

  initThemeSwitcher(); // 添加主题切换初始化

  initBrightnessControl(); // 添加亮度控制初始化

  initVolumeControls(); // 添加音量控制初始化

  initSmartInteractionInputEvents(); // 初始化文本输入相关的事件
  
  // 只在桌面端启用卡片悬浮特效
  if (window.innerWidth > 768) {
    initCardHoverEffects();
  }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initDriverPage);

// 导出函数供其他模块使用
export { initMusicPlayer, initTabSwitching };