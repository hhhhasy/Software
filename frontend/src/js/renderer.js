// 会话状态管理
const session = {
  set: (key, value) => localStorage.setItem(key, JSON.stringify(value)),
  get: (key) => JSON.parse(localStorage.getItem(key)),
  clear: () => localStorage.clear()
}

// 语音录制功能
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

async function toggleRecording() {
  const voiceBtn = document.querySelector('.voice-btn');
  
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

// 登录表单处理
document.getElementById('loginForm')?.addEventListener('submit', async (e) => {
  e.preventDefault()
  
  const username = document.getElementById('username').value
  const password = document.getElementById('password').value

  try {
    const response = await fetch('http://localhost:8000/api/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password})
    })

    if (!response.ok) throw await response.json()
    
    const { role } = await response.json()
    session.set('currentUser', { username, role })
    
    window.location.href = role === 'admin' ? 'admin.html' : 'home.html'
  } catch (err) {
    alert(err.detail || '登录失败')
  }
})

// 注册表单处理
document.getElementById('registerForm')?.addEventListener('submit', async (e) => {
  e.preventDefault()
  
  const newUser = {
    username: document.getElementById('newUsername').value,
    password: document.getElementById('newPassword').value,
    confirm_password: document.getElementById('confirmPassword').value
  }

  try {
    const response = await fetch('http://localhost:8000/api/register', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(newUser)
    })

    if (!response.ok) throw await response.json()
    
    alert('注册成功，请登录')
    window.location.href = '../views/login.html'
  } catch (err) {
    alert(err.detail || '注册失败')
  }
})

async function processVideo() {
  try {
      // // 1. 请求摄像头权限并立即停止，检测可用性
      // const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      // const track = stream.getVideoTracks()[0];
      // track.stop();

      // 2. 向后端发起 POST 请求
      const response = await fetch('http://localhost:8000/api/process-video', { method: 'POST' });
      if (!response.ok) {
          // 如果状态码不是 2xx，抛出后台返回的 JSON 作为错误
          throw await response.json();
      }
      const data = await response.json();

      // 3. 根据返回结果给出提示
      if (data.message) {
          alert('视频处理完成');
      }
  } catch (err) {
      console.error('视频处理错误:', err);
      alert('视频处理失败: ' + (err.detail || '服务器错误'));
  }
}


async function processGesture() {
  try {
      // // 1. 请求摄像头权限并立即停止，检测可用性
      // const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      // const track = stream.getVideoTracks()[0];
      // track.stop();

      // 2. 向后端发起 POST 请求
      const response = await fetch('http://localhost:8000/api/process-gesture', { method: 'POST' });
      if (!response.ok) {
          // 如果状态码不是 2xx，抛出后台返回的 JSON 作为错误
          throw await response.json();
      }
      const data = await response.json();
      
      const audio   = document.getElementById('audioTrack');
      const stopBtn = document.getElementById('stopBtn');

      // 3. 根据返回结果给出提示
      
      if (data.gesture) {
          switch (data.gesture) {
               case 'fist':
                  stopBtn.click();
                  alert('检测到拳，音乐停止！');
                  break;
              case 'OK':
                  alert('检测到ok！');
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

