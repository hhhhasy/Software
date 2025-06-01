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
      if (window.updatePassengerMusicUI) { // 检查函数是否存在
        window.updatePassengerMusicUI(true); // 更新乘客界面的播放按钮状态
      }
      if (window.updateDriverMusicPlayerUI) { // 检查函数是否存在
        window.updateDriverMusicPlayerUI(true); // 更新驾驶员界面的播放按钮状态
      }
    } else {
      showError("找不到音频播放器");
    }
  } 
  else if (commandText.includes("音乐播放已暂停") || commandText.toLowerCase().includes("stop music") || commandText.toLowerCase().includes("pause music")) {
    const audio = document.getElementById('audioTrack');
    if (audio) {
      audio.pause();
      showSuccess("🎵 已暂停音乐");
      if (window.updatePassengerMusicUI) { // 检查函数是否存在
        window.updatePassengerMusicUI(false); // 更新乘客界面的播放按钮状态
      }
      if (window.updateDriverMusicPlayerUI) { // 检查函数是否存在
        window.updateDriverMusicPlayerUI(false); // 更新驾驶员界面的播放按钮状态
      }
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
          if(text.trim().includes('警报已解除')) {
            if (window.stopAppScreenFlash) {
              window.stopAppScreenFlash();
            }
            processVideo();
          }
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
      if(voiceBtn) voiceBtn.textContent = '录音中...';

      // 10秒后自动停止
      setTimeout(() => {
        if (isRecording && mediaRecorder && mediaRecorder.state === "recording") {
          mediaRecorder.stop();
          if(currentStream) currentStream.getTracks().forEach(track => track.stop());
          isRecording = false;
          if(voiceBtn) voiceBtn.textContent = '🎤 语音指令输入';
          showSuccess('录音已自动停止');
        }
      }, 5000);

    } catch (err) {
      hideLoading();
      showError('无法访问麦克风: ' + err.message);
      console.error('录音错误:', err);
      isRecording = false; // 确保状态被重置
      if(voiceBtn) voiceBtn.textContent = '🎤 语音指令输入'; // 恢复按钮文本
    }
  }
  // else {
  //   if (mediaRecorder && mediaRecorder.state === "recording") {
  //       mediaRecorder.stop();
  //   }
  //   if(currentStream) currentStream.getTracks().forEach(track => track.stop()); // 释放麦克风
  //   isRecording = false;
  //   if(voiceBtn) voiceBtn.textContent = '🎤 语音指令输入';
  //   hideLoading(); // 如果之前有加载指示，确保隐藏
  // }
}

// 函数：处理来自后端的响应 (被文本处理共用)
async function handleBackendResponse(responseData, recognizedCommandText = "") {
  const aiResponseText = responseData.text;
  const commandUsed = recognizedCommandText || responseData.command; // 如果是文本输入，responseData.command可能就是输入的文本

  const aiResponseOutput = document.querySelector('#textInput'); // 在driver.html中是 #textInput
  const lastCommandDisplay = document.getElementById('lastCommandDisplay'); // 在driver.html中新增的

  if (aiResponseOutput) aiResponseOutput.textContent = aiResponseText;
  if (lastCommandDisplay) lastCommandDisplay.textContent = `上次命令: ${commandUsed}`;

  // 根据后端返回的alert状态控制闪光灯
  if (responseData.hasOwnProperty('alert')) {
    if (responseData.alert) {
      isDriverAlertActive = true;
      if (window.startAppScreenFlash) window.startAppScreenFlash(0, 600, 'red');
    } else {
      isDriverAlertActive = false;
      if (window.stopAppScreenFlash) window.stopAppScreenFlash();
    }
  }
  if (aiResponseText.trim().includes('警报已解除')) {
    if (window.stopAppScreenFlash) {
      window.stopAppScreenFlash();
    }
  }

  // 调用通用的前端命令处理逻辑
  handleVoiceCommand(aiResponseText); // 用AI的回复文本来触发前端动作（如音乐播放）
}


// 新增：处理文本输入指令的函数
async function processTextInput(commandText) {
  if (!commandText || commandText.trim() === "") {
    showError("指令不能为空！");
    return;
  }

  const aiResponseOutput = document.querySelector('#textInput'); // AI回复显示区域
  const lastCommandDisplay = document.getElementById('lastCommandDisplay'); // 上次命令显示
  const commandTextInputField = document.getElementById('commandTextInput'); // 文本输入框本身


  showLoading('正在处理指令...');
  if (lastCommandDisplay) lastCommandDisplay.textContent = `发送命令: ${commandText}`;
  if (commandTextInputField) commandTextInputField.value = ""; // 清空输入框

  try {
    const currentUser = session.get('currentUser');
    const formData = new FormData(); // 后端 /api/speech-to-text 期望 FormData
    formData.append('text_command', commandText); // 发送文本指令

    const response = await fetch('http://localhost:8000/api/text-command', {
      method: 'POST',
      headers: { 'X-User-ID': currentUser?.id },
      body: formData
    });

    if (!response.ok) throw await response.json();

    const responseData = await response.json();
    showSuccess('指令已发送'); // 或等待后端确认
    await handleBackendResponse(responseData, commandText); // 使用公共函数处理响应

  } catch (err) {
    showError('指令处理失败: ' + (err.detail || err.message || '服务器错误'));
    if (aiResponseOutput) aiResponseOutput.textContent = '处理失败，请重试。';
    // 确保出错时也尝试停止闪光灯
    if (isDriverAlertActive && window.stopAppScreenFlash) window.stopAppScreenFlash();
  } finally {
    hideLoading();
  }
}

// 处理视频识别
async function processVideo() {
  showLoading('正在处理视频...');
  try {
    const currentUser = session.get('currentUser');
    const response = await fetch('http://localhost:8000/api/process-video', {
      method: 'POST',
      headers: { 'X-User-ID': currentUser?.id }
    });
    const data = await response.json();

    if (data.alert) {
      showError(data.message + ' 警报已触发，请说“解除警报”');
      if (window.startAppScreenFlash) { // 检查函数是否存在
        window.startAppScreenFlash(0, 600, 'red'); // 一直闪烁，每0.6秒闪一次，红色
      }
      //toggleRecording(); // 自动激活语音识别
    } else {
      showSuccess(data.message || '视频处理请求已发送');
      if (window.stopAppScreenFlash) { // 检查函数是否存在
        window.stopAppScreenFlash(); // 如果警报解除或未触发警报，确保停止闪光
      }
    }
  } catch (err) {
    showError('视频处理失败: ' + (err.detail || '服务器错误'));
    console.error('视频处理错误:', err);
    if (window.stopAppScreenFlash) { // 出错时也尝试停止闪光
      window.stopAppScreenFlash();
    }
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
    if (data.resp_text.trim().includes('警报已解除')) {
      // processVideo();
      if (window.stopAppScreenFlash) {
        window.stopAppScreenFlash();
      }
    }
    if (data.gesture) {
      let gestureMessage = '未知手势';
      switch (data.gesture) {
        case 'Closed_Fist':
          document.getElementById('audioTrack')?.pause();
          gestureMessage = '检测到拳头，音乐已暂停！';
          break;
        case 'Victory':
          gestureMessage = '检测到胜利手势！';
          break;
        case 'Thumb_Up':
          gestureMessage = '检测到竖起大拇指！';
          break; 
        case 'Open_Palm':
          gestureMessage = '检测到张开手掌！';
          break;
        case 'Pointing_Up':
          gestureMessage = '检测到指向上方！';
          break;
        case 'Thumb_Down':
          gestureMessage = '检测到反对手势！';
          break;
        case 'ILoveYou':
          gestureMessage = '检测到爱你手势！';
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

  // 将 processTextInput 挂载到 window，以便 driver.js 可以调用
  window.processDriverTextInput = processTextInput;
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initMultimedia);

// 导出函数供其他模块使用
export { toggleRecording, processVideo, processGesture };