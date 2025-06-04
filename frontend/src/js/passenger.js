import session from '../utils/session.js';
import { showError, showSuccess } from '../utils/uiUtils.js'; // 乘客界面可能不需要showLoading，除非有特定耗时操作

// 页面初始化只做登出按钮绑定，音乐播放/语音等交由模块统一管理
function initLogout() {
    const logoutBtn = document.getElementById('logout');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            session.clear();
            window.location.replace('login.html');
        });
    }
}

function initMusicPlayer() {
    const audio = document.getElementById('audioTrack');
    const playBtn = document.getElementById('playPauseBtn');
    const ppIcon = document.getElementById('ppIcon');
    const musicIcon = document.getElementById('musicIcon');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const progressBar = document.querySelector('.music-player .progress-bar'); // 用于获取进度条元素
    const currentTimeDisplay = document.querySelector('.music-player .time-current'); // 用于获取当前时间显示元素
    const totalTimeDisplay = document.querySelector('.music-player .time-total'); // 用于获取总时间显示元素

    if (!audio || !playBtn || !ppIcon || !musicIcon || !progressBar || !currentTimeDisplay || !totalTimeDisplay) {
        console.warn('音乐播放器部分元素未找到，功能可能不完整。');
        return;
    }

    // 简单播放列表
    const playlist = [
        '../assets/audio/song1.mp3',
        '../assets/audio/song2.mp3',
    ];
    let currentIndex = 0;

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
            musicIcon.classList.add('playing');
            ppIcon.classList.remove('fa-play');
            ppIcon.classList.add('fa-pause');
        } else {
            musicIcon.classList.remove('playing');
            ppIcon.classList.remove('fa-pause');
            ppIcon.classList.add('fa-play');
        }
    }
    
    function updateProgress() {
        if (!audio.duration) return;
        const percent = (audio.currentTime / audio.duration) * 100;
        progressBar.style.width = `${percent}%`;

        const currentMinutes = Math.floor(audio.currentTime / 60);
        const currentSeconds = Math.floor(audio.currentTime % 60).toString().padStart(2, '0');
        currentTimeDisplay.textContent = `${currentMinutes}:${currentSeconds}`;


        const totalMinutes = Math.floor(audio.duration / 60);
        const totalSeconds = Math.floor(audio.duration % 60).toString().padStart(2, '0');
        totalTimeDisplay.textContent = `${totalMinutes}:${totalSeconds}`;
    }

    playBtn?.addEventListener('click', () => {
        if (audio.paused) {
            audio.play().then(() => {
                updateUI(true);
                showSuccess('音乐已开始播放', 2000); // 2秒后自动消失
            }).catch(error => {
                showError('播放失败: ' + error.message);
                updateUI(false);
            });
        } else {
            audio.pause();
            updateUI(false);
            showSuccess('音乐已暂停', 2000);
        }
    });

    musicIcon?.addEventListener('click', () => playBtn.click());

    prevBtn?.addEventListener('click', () => {
        loadTrack(currentIndex - 1);
        audio.play().then(() => updateUI(true)).catch(error => showError('播放失败: ' + error.message));
    });
    nextBtn?.addEventListener('click', () => {
        loadTrack(currentIndex + 1);
        audio.play().then(() => updateUI(true)).catch(error => showError('播放失败: ' + error.message));
    });
    audio?.addEventListener('ended', () => {
        updateUI(false);
        showSuccess('当前歌曲播放完毕', 2000);
        // 可选择自动播放下一首
        // nextBtn.click(); 
    });
    audio?.addEventListener('timeupdate', updateProgress);
    audio?.addEventListener('loadedmetadata', updateProgress); // 确保音频元数据加载后更新总时间

    loadTrack(currentIndex);
}

// 将 updateUI 函数暴露给其他模块，例如 multimedia.js，以便语音控制可以更新播放器状态
// 注意：这需要 passenger.js 也采用模块化的方式，或者通过全局变量等方式暴露
// 假设 passenger.js 已经是 ES 模块，可以这样导出：
// export { updateUI }; // 如果 passenger.js 不是模块，需要其他方式，或者直接在 multimedia.js 中操作 DOM

// 确保 updateUI 可以在模块外部被调用
window.updatePassengerMusicUI = (isPlaying) => {
    const ppIcon = document.getElementById('ppIcon');
    const musicIcon = document.getElementById('musicIcon');
    if (!ppIcon || !musicIcon) return;

    if (isPlaying) {
        musicIcon.classList.add('playing');
        ppIcon.classList.remove('fa-play');
        ppIcon.classList.add('fa-pause');
    } else {
        musicIcon.classList.remove('playing');
        ppIcon.classList.remove('fa-pause');
        ppIcon.classList.add('fa-play');
    }
};

function initPassengerPage() {
    initLogout();
    initMusicPlayer();
    // 其余交互由auth-check.js和multimedia.js负责
}

document.addEventListener('DOMContentLoaded', initPassengerPage);
