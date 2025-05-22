import session from '../utils/session.js';

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
            audio.play();
            updateUI(true);
        } else {
            audio.pause();
            updateUI(false);
        }
    });

    musicIcon?.addEventListener('click', () => playBtn.click());

    prevBtn?.addEventListener('click', () => {
        loadTrack(currentIndex - 1);
        audio.play();
        updateUI(true);
    });
    nextBtn?.addEventListener('click', () => {
        loadTrack(currentIndex + 1);
        audio.play();
        updateUI(true);
    });
    audio?.addEventListener('ended', () => updateUI(false));
    loadTrack(currentIndex);
}

function initPassengerPage() {
    initLogout();
    initMusicPlayer();
    // 其余交互由auth-check.js和multimedia.js负责
}

document.addEventListener('DOMContentLoaded', initPassengerPage);
