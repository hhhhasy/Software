/**
 * 多媒体处理模块 - 语音、视频和手势处理
 */
// 会话状态管理
import session from '../utils/session.js';
import { showLoading, hideLoading, showError, showSuccess } from '../utils/uiUtils.js';

function handleVoiceCommand(commandText) {
  commandText = commandText.trim();

  if (commandText.includes("为您播放默认播放列表") || commandText.toLowerCase().includes("play music")) {
    const audio = document.getElementById('audioTrack');
    if (audio) {
      audio.play();
      showSuccess("🎵 已播放音乐");
    } else {
      showError("找不到音频播放器");
    }
  } 
  else if (commandText.includes("音乐播放已暂停") || commandText.toLowerCase().includes("stop music") || commandText.toLowerCase().includes("pause music")) {
    const audio = document.getElementById('audioTrack');
    if (audio) {
      audio.pause();
      showSuccess("🎵 已暂停音乐");
    } else {
      showError("找不到音频播放器");
    }
  }
  // 可以根据需要添加更多语音命令处理
}

// 语音录制变量
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let currentStream = null; // 添加此变量用于后续关闭麦克风

// 切换语音录制状态
async function toggleRecording() {
  const voiceBtn = document.querySelector('#voiceBtn');

  if (!isRecording) {
    showLoading('正在准备录音...');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      currentStream = stream; // 保存 stream 到全局变量
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        audioChunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        hideLoading(); // 隐藏“准备录音”的加载
        showLoading('正在识别语音...');
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        try {
          const currentUser = session.get('currentUser')
          const response = await fetch('http://localhost:8000/api/speech-to-text', {
            method: 'POST',
            headers: {
              'X-User-ID': currentUser?.id   // 从登录状态中获得
            },

            body: formData
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw errorData; // 抛出后端返回的错误信息
          }

          const { command, text } = await response.json();
          const textInput = document.querySelector('#textInput');
          if(textInput) textInput.textContent = text;
          showSuccess('语音识别成功: ' + text);
          handleVoiceCommand(text);
        } catch (err) {
          showError('语音识别失败: ' + (err.detail || '服务器错误'));
          console.error('语音识别错误:', err);
        } finally {
          hideLoading();
        }
      };

      mediaRecorder.start();
      hideLoading(); // 隐藏“准备录音”的加载
      isRecording = true;
      if(voiceBtn) voiceBtn.textContent = '⏹ 停止录音';

      // 10秒后自动停止
      setTimeout(() => {
        if (isRecording && mediaRecorder && mediaRecorder.state === "recording") {
          mediaRecorder.stop();
          if(currentStream) currentStream.getTracks().forEach(track => track.stop());
          isRecording = false;
          if(voiceBtn) voiceBtn.textContent = '🎤 语音指令输入';
          showSuccess('录音已自动停止');
        }
      }, 10000);

    } catch (err) {
      hideLoading();
      showError('无法访问麦克风: ' + err.message);
      console.error('录音错误:', err);
      isRecording = false; // 确保状态被重置
      if(voiceBtn) voiceBtn.textContent = '🎤 语音指令输入'; // 恢复按钮文本
    }
  } else {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
    if(currentStream) currentStream.getTracks().forEach(track => track.stop()); // 释放麦克风
    isRecording = false;
    if(voiceBtn) voiceBtn.textContent = '🎤 语音指令输入';
    hideLoading(); // 如果之前有加载指示，确保隐藏
  }
}

// 处理视频识别
async function processVideo() {
  showLoading('正在处理视频...');
  try {
    const currentUser = session.get('currentUser')
    const response = await fetch('http://localhost:8000/api/process-video', { 
      method: 'POST',
      headers: {
              'X-User-ID': currentUser?.id   // 从登录状态中获得
            }
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw errorData;
    }
    const data = await response.json();

    if (data.message) {
      showSuccess('视频处理完成: ' + data.message);
    } else {
      showSuccess('视频处理请求已发送');
    }
  } catch (err) {
    showError('视频处理失败: ' + (err.detail || '服务器错误'));
    console.error('视频处理错误:', err);
  } finally {
    hideLoading();
  }
}

// 处理手势识别
async function processGesture() {
  showLoading('正在识别手势...');
  try {
    const currentUser = session.get('currentUser')
    const response = await fetch('http://localhost:8000/api/process-gesture', { 
      method: 'POST',
      headers: {
              'X-User-ID': currentUser?.id   // 从登录状态中获得
            }
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw errorData;
    }
    const data = await response.json();
    
    if (data.gesture) {
      let gestureMessage = '未知手势';
      switch (data.gesture) {
        case 'fist':
          document.getElementById('audioTrack')?.pause();
          gestureMessage = '检测到拳头，音乐已暂停！';
          break;
        case 'OK':
          gestureMessage = '检测到OK手势！';
          break;
        case 'thumbs_up':
          gestureMessage = '检测到竖起大拇指！';
          break; 
        case 'palm':
          gestureMessage = '检测到张开手掌！';
          break;
      }
      showSuccess(gestureMessage);
    } else if (data.message) { // 有可能后端只返回一个消息
      showSuccess(data.message);
    } else {
      showError('未能识别有效手势');
    }
  } catch (err) {
    showError('手势处理失败: ' + (err.detail || '服务器错误'));
    console.error('手势处理错误:', err);
  } finally {
    hideLoading();
  }
}

// 初始化多媒体功能
function initMultimedia() {
  // 绑定按钮事件
  document.getElementById('voiceBtn')?.addEventListener('click', toggleRecording);
  document.getElementById('videoBtn')?.addEventListener('click', processVideo);
  document.getElementById('gestureBtn')?.addEventListener('click', processGesture);
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initMultimedia);

// 导出函数供其他模块使用
export { toggleRecording, processVideo, processGesture };