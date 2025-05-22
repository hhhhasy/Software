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
    
    if (!audio) {
        // 如果播放器核心元素不存在，则不进行初始化，可选择显示错误
        // showError('音乐播放器未能正确加载。');
        console.warn('音乐播放器核心元素未找到，无法初始化。');
        return;
    }

    // 简单播放列表
    const playlist = [
        '../assets/audio/song1.mp3'
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
    loadTrack(currentIndex);
}

function initPassengerPage() {
    initLogout();
    initMusicPlayer();
    // 其余交互由auth-check.js和multimedia.js负责
}

document.addEventListener('DOMContentLoaded', initPassengerPage);
