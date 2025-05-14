/**
 * 多媒体处理模块 - 语音、视频和手势处理
 */

// 语音录制变量
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// 切换语音录制状态
async function toggleRecording() {
  const voiceBtn = document.querySelector('#voiceBtn');
  
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      
      mediaRecorder.ondataavailable = (e) => {
        audioChunks.push(e.data);
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        
        try {
          const response = await fetch('http://localhost:8000/api/speech-to-text', {
            method: 'POST',
            body: formData
          });
          
          if (!response.ok) throw await response.json();
          
          const { text } = await response.json();
          alert(text);
        } catch (err) {
          console.error('语音识别错误:', err);
          alert('语音识别失败: ' + (err.detail || '服务器错误'));
        }
      };
      
      mediaRecorder.start();
      isRecording = true;
      voiceBtn.textContent = '⏹ 停止录音';
      
      // 5秒后自动停止
      setTimeout(() => {
        if (isRecording) {
          mediaRecorder.stop();
          stream.getTracks().forEach(track => track.stop());
          isRecording = false;
          voiceBtn.textContent = '🎤 语音指令输入';
        }
      }, 5000);
      
    } catch (err) {
      console.error('录音错误:', err);
      alert('无法访问麦克风: ' + err.message);
    }
  } else {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
    voiceBtn.textContent = '🎤 语音指令输入';
  }
}

// 处理视频识别
async function processVideo() {
  try {
    const response = await fetch('http://localhost:8000/api/process-video', { method: 'POST' });
    if (!response.ok) {
      throw await response.json();
    }
    const data = await response.json();

    if (data.message) {
      alert('视频处理完成');
    }
  } catch (err) {
    console.error('视频处理错误:', err);
    alert('视频处理失败: ' + (err.detail || '服务器错误'));
  }
}

// 处理手势识别
async function processGesture() {
  try {
    const response = await fetch('http://localhost:8000/api/process-gesture', { method: 'POST' });
    if (!response.ok) {
      throw await response.json();
    }
    const data = await response.json();
    
    if (data.gesture) {
      switch (data.gesture) {
        case 'fist':
          document.getElementById('stopBtn')?.click();
          alert('检测到拳，音乐停止！');
          break;
        case 'OK':
          alert('检测到OK！');
          break;
        case 'thumbs_up':
          alert('检测到赞！');
          break; 
        case 'palm':
          alert('检测到手展开！');
          break;
      }
    }
  } catch (err) {
    console.error('手势处理错误:', err);
    alert('手势处理失败: ' + (err.detail || '服务器错误'));
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