"""
业务逻辑服务模块
包含语音识别、视频处理和手势识别的核心功能
"""
import os
import time
import logging
import tempfile
import json
import numpy as np
import joblib
import asyncio
import cv2
import pyttsx3
import whisper
import mediapipe as mp
import subprocess
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound
from openvino.runtime import Core
from fastapi import HTTPException, status
from models import UserMemory
from zhipu import call_zhipu_chat
from models import UserPreference
from mediapipe.tasks.python.vision import GestureRecognizerResult

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)  # 确保模型目录存在

# 日志配置
logger = logging.getLogger("services")
mmlog = logging.getLogger("multimodal")

# 全局变量
whisper_model = None
tts_engine = pyttsx3.init()
alert_active = False
alert_task = None

# ========== 多模态日志工具 ==========
def log_multimodal(user_id: int, modality: str, content: str, response: str):
    """
    结构化写多模态交互日志
    - user_id : 用户 ID（0 表示匿名）
    - modality: speech / gesture / vision
    - content : 用户输入 ("播放音乐", "fist"…)
    - response: 系统反馈 ("为您播放默认播放列表"…)
    """
    mmlog.info(
        f"user:{user_id} | modality:{modality} | content:{content} | response:{response}"
    )

# ========== 语音服务 ==========
def get_whisper_model():
    """延迟加载Whisper模型"""
    global whisper_model
    if whisper_model is None:
        try:
            logger.info("正在加载Whisper模型...")
            whisper_model = whisper.load_model("small")  # 使用small模型提高速度
            logger.info("Whisper模型加载完成")
        except Exception as e:
            logger.error(f"加载Whisper模型失败: {e}")
            raise RuntimeError(f"无法加载语音识别模型: {e}")
    return whisper_model

def convert_to_whisper_format(input_path, output_path=None):
    """
    使用 ffmpeg 将音频文件转换为 Whisper 推荐的格式（16kHz、mono、wav）
    
    参数:
    - input_path: 原始音频文件路径
    - output_path: 转换后文件保存路径，默认在原路径后添加 '_converted.wav'
    
    返回:
    - output_path: 转换后的音频文件路径
    """
    if output_path is None:
        output_path = input_path.replace('.wav', '_converted.wav')

    # 确保输入文件存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到音频文件: {input_path}")

    command = [
        'ffmpeg', '-y',                # -y 自动覆盖
        '-i', input_path,              # 输入文件
        '-ar', '16000',                # 设置采样率 16kHz
        '-ac', '1',                    # 设置为单声道
        '-f', 'wav',                   # 输出格式为 WAV
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"音频转换失败: {e}")
    return output_path

async def process_speech(audio_path: str, user_id: int, db: Session) -> Dict[str, str]:
    """
    处理语音并返回响应
    
    参数:
    - audio_path: 音频文件路径
    - user_id: 用户ID
    - db: 数据库会话
    
    返回:
    - Dict[str, str]: 包含识别结果和响应文本
    """
    global alert_active
    
    try:
        # 获取模型并进行语音识别
        model = get_whisper_model()
        converted_path = convert_to_whisper_format(audio_path)
        result = model.transcribe(converted_path, language='zh')
        recognized_text = result["text"]
        logger.info(f"语音识别结果: {recognized_text}")

        # 清理临时文件
        try:
            if os.path.exists(converted_path) and converted_path != audio_path:
                os.unlink(converted_path)
        except Exception as e:
            logger.warning(f"删除转换后的音频文件失败: {str(e)}")

        # 解除警报关键词
        if alert_active:
            if "解除警报" in recognized_text or "取消警报" in recognized_text:
                alert_active = False
                try:
                    tts_engine.say("警报已解除")
                    tts_engine.runAndWait()
                except Exception as e:
                    logger.warning(f"语音输出失败: {str(e)}")
                return {"command": recognized_text, "text": "警报已解除"}
            else:
                return {"command": recognized_text, "text": "警报未解除"}

        # 检查是否匹配预设指令
        response = get_command_response(recognized_text)
        if response:
            try:
                # 使用语音引擎进行语音输出
                tts_engine.say(response)
                tts_engine.runAndWait()
            except Exception as e:
                logger.warning(f"语音输出失败: {str(e)}")
            
            log_multimodal(user_id, "speech", recognized_text, response)
            await update_common_commands(user_id, recognized_text, db)
            return {
                "command": recognized_text,
                "text": response
            }
        
        # 如果未匹配任何预设指令，则调用大模型进行回答
        logger.info(f"用户 {user_id} 发起大模型请求")
        
        # 使用 zhipu_chat 的逻辑来读取和更新对话历史
        try:
            memory = db.query(UserMemory).filter(UserMemory.user_id == user_id).one()
            history = json.loads(memory.content)
        except NoResultFound:
            history = []
            memory = UserMemory(user_id=user_id, content="[]")

        # 将当前请求的对话追加到历史中
        history.append({"role": "user", "content": recognized_text})

        MAX_HISTORY = 9
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]

        # 调用大模型
        ai_reply = await call_zhipu_chat(history)

        # 添加 assistant 回复
        history.append({"role": "assistant", "content": ai_reply})

        # 写回数据库
        memory.content = json.dumps(history, ensure_ascii=False)
        db.merge(memory)
        db.commit()
        logger.info(f"用户 {user_id} 对话历史已更新")
        
        # 语音播报大模型回答
        try:
            tts_engine.say(ai_reply)
            tts_engine.runAndWait()
        except Exception as e:
            logger.warning(f"语音输出失败: {str(e)}")

        log_multimodal(user_id, "speech", recognized_text, ai_reply)
        return {
            "command": recognized_text,
            "text": ai_reply
        }
            
    except Exception as e:
        logger.error(f"语音识别错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"语音识别错误: {str(e)}"
        )

def get_command_response(text: str) -> Optional[str]:
    """根据识别文本获取对应的指令响应"""
    command_mapping = {
        "打开空调": "已经打开空调",
        "关闭空调": "已经关闭空调",
        "把温度调高": "已经调高温度",
        "调低温度": "已经调低温度",
        "把风速调到中档": "风速已调至中档",
        "最大风速": "已将风速调至最大档位",
        "把风向调到吹向前排": "风向已调整为前排吹风模式",
        "打开内循环": "已切换到内循环模式",
        "切换外循环": "已切换到外循环模式",
        "开启前挡风玻璃除雾": "前挡除雾功能已启动",
        "关闭后挡风玻璃除雾": "已关闭后挡除雾",
        "打开司机座椅加热": "司机座椅加热已开启",
        "关闭副驾驶座椅加热": "已关闭副驾驶座椅加热",
        "播放音乐": "为您播放默认播放列表",
        "播放周杰伦的歌": "正在播放周杰伦的热门歌曲",
        "暂停音乐": "音乐播放已暂停",
        "下一首歌": "已切换到下一首歌曲",
        "上一首歌": "已播放上一首歌曲",
        "导航到最近的加油站": "已规划到最近加油站的路线，开始导航",
        "取消导航": "导航已取消",
        "查询剩余油量": "当前油箱剩余约40%",
        "打开左后车窗": "左后车窗已打开",
        "关闭所有车窗": "所有车窗已关闭并锁定",
        "检查胎压": "所有轮胎胎压正常，前轮36 PSI，后轮35 PSI",
        "切换到运动模式": "已切换至运动模式",
        "切换到舒适模式": "已切换至舒适模式"
    }
    
    for command, response in command_mapping.items():
        if command in text:
            logger.info(f"匹配到指令: {command}")
            return response
    
    return None

# ========== 持续警报功能（支持警报升级） ==========
async def continuous_warning():
    """持续发出警告音频，达到一定次数后升级警报内容"""
    global alert_active
    num = 0
    upgrade_threshold = 5  # 每播报 5 次后升级一次警报
    max_repeats = 50       # 最大播报次数限制

    while alert_active and num < max_repeats:
        num += 1

        # 判断当前应该使用哪种警报内容
        if num <= upgrade_threshold:
            message = "注意力偏离，请集中注意力！如需解除警报，请说'解除警报' 或者 做 '胜利' 手势。"
        elif num <= upgrade_threshold * 2:
            message = "警告！您仍未集中注意力！如需解除，请立即作出回应！"
        else:
            message = "严重警告！请立即集中注意力，否则系统将记录此行为！"

        try:
            tts_engine.say(message)
            tts_engine.runAndWait()
        except Exception as e:
            logger.warning(f"持续警报播报失败: {str(e)}")
            break

        await asyncio.sleep(2)


# ========== 视频处理服务 ==========
# 模型路径
VINO_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intel")

# 加载 OpenVINO 模型
def load_openvino_models():
    """加载所有OpenVINO模型"""
    try:
        core = Core()
        
        # 人脸检测模型
        face_det_model = core.read_model(f"{VINO_MODEL_DIR}/face-detection-adas-0001/FP16/face-detection-adas-0001.xml")
        face_det = core.compile_model(face_det_model, "CPU")
        face_output_layer = face_det.output(0)
        
        # 面部特征点模型
        landmarks_model = core.read_model(f"{VINO_MODEL_DIR}/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml")
        landmarks_det = core.compile_model(landmarks_model, "CPU")
        landmarks_output_layer = landmarks_det.output(0)
        
        # 头部姿态估计模型
        head_pose_model = core.read_model(f"{VINO_MODEL_DIR}/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml")
        head_pose_det = core.compile_model(head_pose_model, "CPU")
        
        # 视线估计模型
        gaze_model = core.read_model(f"{VINO_MODEL_DIR}/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml")
        gaze_det = core.compile_model(gaze_model, "CPU")
        
        return {
            "core": core,
            "face_det": face_det,
            "face_output_layer": face_output_layer,
            "landmarks_det": landmarks_det,
            "landmarks_output_layer": landmarks_output_layer,
            "head_pose_det": head_pose_det,
            "gaze_det": gaze_det
        }
    except Exception as e:
        logger.error(f"加载OpenVINO模型失败: {str(e)}")
        raise RuntimeError(f"无法加载OpenVINO模型: {e}")

def preprocess_for_openvino(image, target_shape):
    """预处理图像用于OpenVINO推理"""
    h, w = target_shape
    resized = cv2.resize(image, (w, h))
    return resized.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)

def safe_crop(image, center_x, center_y, size):
    """安全裁剪图像区域，确保在图像边界内"""
    h, w = image.shape[:2]
    half = size // 2
    x1 = max(int(center_x) - half, 0)
    y1 = max(int(center_y) - half, 0)
    x2 = min(int(center_x) + half, w)
    y2 = min(int(center_y) + half, h)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return None
    return cropped

def compute_ear(eye_points, landmarks, img_w, img_h):
    """计算眼睛纵横比(EAR)"""
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_points]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

async def process_video():
    """处理视频并监测驾驶员状态"""
    global alert_active, alert_task
    
    logger.info("启动驾驶员状态监测")
    mp_face_mesh = None
    face_mesh = None
    cap = None
    
    try:
        # 加载模型
        models = load_openvino_models()
        
        # 初始化 MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法访问摄像头"
            )

        # 阈值和索引定义
        LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
        EAR_THRESH = 0.25
        CLOSED_SEC_THRESHOLD = 5.0

        # 打哈欠
        UPPER_LIP_IDX = 13
        LOWER_LIP_IDX = 14
        LEFT_MOUTH_IDX = 78
        RIGHT_MOUTH_IDX = 308
        YAWN_MAR_THRESH = 0.5
        YAWN_FRAMES_THRESH = 15

        # 点头/摇头
        nod_counter = 0
        shake_counter = 0
        NOD_THRESH_COUNT = 5
        SHAKE_THRESH_COUNT = 5
        PITCH_THRESH_DEG = 10
        YAW_THRESH_DEG = 10

        # 状态变量
        eye_closed_start = None
        gaze_off_start_time = None  # 注意力偏离起始时间
        yawn_frames = 0
        warnings = []
        start_time = time.time()
        
        # 最多运行30秒或者收集到3个警告
        while (time.time() - start_time < 30) and len(warnings) < 3:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # OpenVINO 人脸检测
            face_input = preprocess_for_openvino(frame, (384, 672))  # 模型默认输入大小
            result = models["face_det"](face_input)[models["face_output_layer"]]
            for det in result[0][0]:
                if det[2] > 0.6:
                    xmin = int(det[3] * w)
                    ymin = int(det[4] * h)
                    xmax = int(det[5] * w)
                    ymax = int(det[6] * h)
                    face_roi = frame[ymin:ymax, xmin:xmax]

                    # 面部关键点
                    lm_input = preprocess_for_openvino(face_roi, (60, 60))
                    lm_result = models["landmarks_det"](lm_input)[models["landmarks_output_layer"]][0]
                    left_eye = lm_result[0:2] * [face_roi.shape[1], face_roi.shape[0]]
                    right_eye = lm_result[2:4] * [face_roi.shape[1], face_roi.shape[0]]

                    # 安全裁剪眼睛区域
                    left_eye_img = safe_crop(face_roi, left_eye[0], left_eye[1], 30)
                    right_eye_img = safe_crop(face_roi, right_eye[0], right_eye[1], 30)

                    # 如果任意一个为空则跳过本轮
                    if left_eye_img is None or right_eye_img is None:
                        logger.warning("无法提取眼睛图像，跳过该帧")
                        continue

                    # 头部姿态
                    head_pose_input = preprocess_for_openvino(face_roi, (60, 60))
                    yaw = models["head_pose_det"](head_pose_input)[models["head_pose_det"].output("angle_y_fc")][0][0]
                    pitch = models["head_pose_det"](head_pose_input)[models["head_pose_det"].output("angle_p_fc")][0][0]
                    roll = models["head_pose_det"](head_pose_input)[models["head_pose_det"].output("angle_r_fc")][0][0]

                    # 视线估计
                    gaze_input = {
                        "left_eye_image": preprocess_for_openvino(left_eye_img, (60, 60)),
                        "right_eye_image": preprocess_for_openvino(right_eye_img, (60, 60)),
                        "head_pose_angles": np.array([[yaw, pitch, roll]], dtype=np.float32)
                    }
                    gaze_vector = models["gaze_det"](gaze_input)[models["gaze_det"].output(0)][0]

                    # 偏离判断（阈值可调）
                    if abs(gaze_vector[0]) > 0.4 or abs(gaze_vector[1]) > 0.4:
                        # 如果是刚开始偏离，记录时间
                        if gaze_off_start_time is None:
                            gaze_off_start_time = time.time()
                        elif time.time() - gaze_off_start_time > 0.75:  # 持续偏离超过1秒
                            warning = "注意力偏离前方，请集中注意力！"
                            if not alert_active:
                                alert_active = True
                                if alert_task is None or alert_task.done():
                                    alert_task = asyncio.create_task(continuous_warning())
                            return {"message": warning, "alert": True}
                    else:
                        # 注意力恢复，重置状态
                        gaze_off_start_time = None
                        alert_active = False

                    # 画框显示
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"Gaze: ({gaze_vector[0]:.2f}, {gaze_vector[1]:.2f})", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # 闭眼检测
                left_ear = compute_ear(LEFT_EYE_IDX, lm, w, h)
                right_ear = compute_ear(RIGHT_EYE_IDX, lm, w, h)
                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESH:
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    else:
                        elapsed = time.time() - eye_closed_start
                        if elapsed >= CLOSED_SEC_THRESHOLD:
                            warning = "请注意，您已经闭眼超过5秒钟！"
                            if warning not in warnings:
                                warnings.append(warning)
                                try:
                                    tts_engine.say(warning)
                                    tts_engine.runAndWait()
                                except Exception as e:
                                    logger.warning(f"语音输出失败: {str(e)}")
                else:
                    eye_closed_start = None

                # 打哈欠检测
                ul = np.array([lm[UPPER_LIP_IDX].x * w, lm[UPPER_LIP_IDX].y * h])
                ll = np.array([lm[LOWER_LIP_IDX].x * w, lm[LOWER_LIP_IDX].y * h])
                lm_pt = np.array([lm[LEFT_MOUTH_IDX].x * w, lm[LEFT_MOUTH_IDX].y * h])
                rm_pt = np.array([lm[RIGHT_MOUTH_IDX].x * w, lm[RIGHT_MOUTH_IDX].y * h])

                mar = np.linalg.norm(ul - ll) / np.linalg.norm(lm_pt - rm_pt)
                if mar > YAWN_MAR_THRESH:
                    yawn_frames += 1
                else:
                    yawn_frames = 0

                if yawn_frames >= YAWN_FRAMES_THRESH:
                    warning = "请注意，您正在打哈欠！"
                    if warning not in warnings:
                        warnings.append(warning)
                        try:
                            tts_engine.say(warning)
                            tts_engine.runAndWait()
                        except Exception as e:
                            logger.warning(f"语音输出失败: {str(e)}")

                # 点头/摇头检测
                face_2d, face_3d = [], []
                for idx, lm_pt in enumerate(lm):
                    if idx in [1, 33, 263, 61, 291, 199]:
                        x, y = int(lm_pt.x * w), int(lm_pt.y * h)
                        if idx == 1:
                            nose_2d = (x, y)
                            nose_3d = np.array([x, y, lm_pt.z * 3000], dtype=np.float64)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm_pt.z])
                
                if len(face_2d) > 0:  # 确保有足够的点
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # 相机参数
                    focal_length = w
                    cam_matrix = np.array([[focal_length, 0, w/2],
                                        [0, focal_length, h/2],
                                        [0, 0, 1]], dtype=np.float64)
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    try:
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        if success:
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                            pitch = angles[0] * 360
                            yaw = angles[1] * 360

                            # 累计计数
                            if pitch < -PITCH_THRESH_DEG:
                                nod_counter += 1
                            else:
                                nod_counter = 0

                            if abs(yaw) > YAW_THRESH_DEG:
                                shake_counter += 1
                            else:
                                shake_counter = 0

                            if nod_counter >= NOD_THRESH_COUNT:
                                warning = "您的动作状态为确认!"
                                if warning not in warnings:
                                    warnings.append(warning)
                                    try:
                                        tts_engine.say(warning)
                                        tts_engine.runAndWait()
                                    except Exception as e:
                                        logger.warning(f"语音输出失败: {str(e)}")

                            if shake_counter >= SHAKE_THRESH_COUNT:
                                warning = "您的动作状态为拒绝!"
                                if warning not in warnings:
                                    warnings.append(warning)
                                    try:
                                        tts_engine.say(warning)
                                        tts_engine.runAndWait()
                                    except Exception as e:
                                        logger.warning(f"语音输出失败: {str(e)}")
                    except Exception as e:
                        logger.warning(f"头部姿势检测计算错误: {str(e)}")
            
            # 显示结果
            cv2.imshow('Head Pose Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # 记录检测到的警告
        if warnings:
            logger.info(f"检测到以下警告: {', '.join(warnings)}")
        else:
            logger.info("未检测到异常情况")
        
        return {"message": "视频处理完成", "alert": False}
    except Exception as e:
        logger.error(f"视频处理错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"视频处理错误: {str(e)}"
        )
    finally:
        # 确保资源释放
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

# ========== 手势识别服务 ==========
import os
import time
import cv2
from fastapi import HTTPException, status
from typing import Dict
from collections import deque, Counter
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    GestureRecognizerResult
)
VisionRunningMode = mp.tasks.vision.RunningMode

async def process_gesture(user_id: int) -> Dict[str, str]:
    """
    使用 MediaPipe GestureRecognizer V2 进行手势识别
    """
    global alert_active
    cap = None
    recognized_label = None
    display_start = None
    resp_text = ""

    try:
        logger.info("启动手势识别")

        # 1. 加载模型文件
        model_path = Path(__file__).parent / "model" / "gesture_recognizer.task"
        if not model_path.exists():
            raise HTTPException(status_code=500, detail="未找到 gesture_recognizer.task 模型文件")
        with open(model_path, "rb") as f:
            model_buffer = f.read()
        logger.info(f"找到模型文件：{model_path}")

        # 2. 滑动窗口配置（调试时先把阈值调低）
        gesture_history = deque(maxlen=5)    # 临时5帧窗口
        stable_threshold = 2                  # 临时2次一致就算稳定

        # 3. 回调函数
        def gesture_callback(result, output_image: mp.Image, timestamp_ms: int):
            nonlocal recognized_label, display_start
            if result.gestures:
                # result.gestures 是一个列表：[[分类1, 分类2...]]
                gesture = result.gestures[0][0].category_name
                gesture_history.append(gesture)
                most_common, freq = Counter(gesture_history).most_common(1)[0]
                logger.debug(f"[回调] 最新一帧分类：{gesture}，滑动窗口统计：{most_common}({freq})")
                if freq >= stable_threshold and recognized_label is None:
                    recognized_label = most_common
                    display_start = time.time()
                    logger.info(f"稳定识别到手势: {recognized_label}")

        # 4. 创建 GestureRecognizer
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_buffer=model_buffer),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=gesture_callback,
            num_hands=1
        )
        recognizer = GestureRecognizer.create_from_options(options)
        logger.info(f"GestureRecognizer 创建成功：{recognizer}")

        # 5. 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="无法访问摄像头")
        logger.info("摄像头打开成功")

        start_time = time.time()

        # 6. 主循环：最多循环10秒，或一旦成功识别就退出
        while (time.time() - start_time < 10) and recognized_label is None:
            ret, frame = cap.read()
            if not ret:
                logger.warning("读取摄像头帧失败，退出循环")
                break

            # ——关键点：BGR 转 RGB——
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = int(time.time() * 1000)

            # 异步识别
            recognizer.recognize_async(mp_image, timestamp)

            # 给后台线程一个小停顿，让它更有机会把回调执行出来
            # 如果你的帧率很低，也可以把下面这行注释掉
            time.sleep(0.005)

            # 如果已经稳定识别到，就把文字画在屏幕上
            if recognized_label:
                cv2.putText(
                    frame,
                    recognized_label,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3
                )

            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                logger.info("用户按下 ESC，退出识别")
                break

            # 如果识别到后显示时间超过1秒，也退出
            if display_start and (time.time() - display_start > 1.0):
                logger.info("已显示结果超过1秒，退出循环")
                break

        # 7. 识别结果映射
        gesture_resp_map = {
            'Closed_Fist': "检测到拳头，音乐已暂停",
            'Thumb_Up': "收到确认",
            'Open_Palm': "检测到手掌张开",
            'Victory': "检测到胜利手势，警报已解除",
            'Pointing_Up': "检测到指向上方手势",
            'Thumb_Down': "检测到反对手势",
            'ILoveYou': "检测到爱你手势"
        }
        resp_text = gesture_resp_map.get(recognized_label, "")

        # 如果识别到，写日志并处理警报逻辑
        if recognized_label:
            log_multimodal(user_id, "gesture", recognized_label, resp_text)
            logger.info(f"user_id={user_id}，识别到手势 {recognized_label}，对应文字：{resp_text}")

            if alert_active and recognized_label == "Victory":
                alert_active = False
                resp_text = "警报已解除"
                try:
                    tts_engine.say("警报已解除")
                    tts_engine.runAndWait()
                except Exception as e:
                    logger.warning(f"语音输出失败: {str(e)}")

        else:
            logger.warning("10秒内没有稳定识别到任何手势")

        return {
            'gesture': recognized_label,
            'resp_text': resp_text
        }

    except Exception as e:
        logger.error(f"手势识别错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"手势识别错误: {str(e)}"
        )
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        # 记得关闭 recognizer 以释放资源
        try:
            recognizer.close()
        except:
            pass

def extract_relative_features(lm):
    """
    以 wrist (0号关键点) 为基准，提取相对位置坐标
    """
    base_x, base_y, base_z = lm.landmark[0].x, lm.landmark[0].y, lm.landmark[0].z
    relative = []
    for p in lm.landmark:
        relative.extend([p.x - base_x, p.y - base_y, p.z - base_z])
    return relative

async def update_common_commands(user_id: int, command_text: str, db: Session):
    try:
        preference = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        if not preference:
            preference = UserPreference(user_id=user_id, common_commands="{}")
            db.add(preference)

        commands_dict = json.loads(preference.common_commands or "{}")
        commands_dict[command_text] = commands_dict.get(command_text, 0) + 1
        preference.common_commands = json.dumps(commands_dict, ensure_ascii=False)
        db.commit()
        logger.info(f"用户 {user_id} 常用指令 '{command_text}' 已更新")
    except Exception as e:
        db.rollback()
        logger.error(f"更新用户 {user_id} 常用指令失败: {e}")

# 用于记录确认模态的函数， "text_input" 也是有效模态
VALID_CONFIRMATION_MODALITIES = ["voice", "gesture", "visual_nod", "text_input"]
async def record_confirmation_modality(user_id: int, modality: str, db: Session):
    if modality not in VALID_CONFIRMATION_MODALITIES:
        logger.warning(f"无效的确认模态: {modality} for user {user_id}")
        return
    try:
        preference = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        if not preference:
            preference = UserPreference(user_id=user_id, interaction_habits="{}")
            db.add(preference)

        habits_dict = {}
        if preference.interaction_habits:
            try:
                habits_dict = json.loads(preference.interaction_habits)
            except json.JSONDecodeError:
                habits_dict = {}

        if "confirmation_modalities" not in habits_dict or not isinstance(habits_dict.get("confirmation_modalities"), dict):
            habits_dict["confirmation_modalities"] = {m: 0 for m in VALID_CONFIRMATION_MODALITIES}
        
        current_counts = habits_dict["confirmation_modalities"]
        current_counts[modality] = current_counts.get(modality, 0) + 1
        
        preference.interaction_habits = json.dumps(habits_dict, ensure_ascii=False)
        db.commit()
        logger.info(f"用户 {user_id} 确认模态 '{modality}' 已记录。当前计数: {current_counts}")
    except Exception as e:
        db.rollback()
        logger.error(f"记录用户 {user_id} 确认模态 '{modality}' 失败: {e}")


async def _apply_command_aliases(user_id: int, text: str, db: Session) -> str:
    """(辅助函数) 应用用户定义的指令别名"""
    try:
        preference = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        if preference and preference.command_aliases:
            aliases_dict = json.loads(preference.command_aliases or "{}")
            if text in aliases_dict:
                original_command = aliases_dict[text]
                logger.info(f"用户 {user_id} 指令别名 '{text}' 转换为 '{original_command}'")
                return original_command
    except Exception as e:
        logger.error(f"应用用户 {user_id} 指令别名失败: {e}")
    return text

async def _process_recognized_text(
    recognized_text: str,
    user_id: int,
    db: Session,
    input_modality: str = "unknown" # "speech", "text_input"
) -> Dict[str, Any]: # 返回值包含 'command', 'text', 'alert'
    """
    核心逻辑：处理已识别的文本（来自语音或直接文本输入）。
    """
    global alert_active
    
    # 1. 应用别名
    processed_text = await _apply_command_aliases(user_id, recognized_text, db)

    current_alert_status = alert_active # 先保存当前状态，用于响应

    # 2. 解除警报关键词处理
    if alert_active:
        if "解除警报" in processed_text or "取消警报" in processed_text:
            alert_active = False
            current_alert_status = False # 更新响应中的状态
            try:
                tts_engine.say("警报已解除")
                tts_engine.runAndWait()
            except Exception as e:
                logger.warning(f"TTS 输出 '警报已解除' 失败: {e}")
            
            await record_confirmation_modality(user_id, input_modality, db) # 记录解除模态
            log_multimodal(user_id, input_modality, recognized_text, "警报已解除")
            await update_common_commands(user_id, recognized_text, db) # 记录常用指令
            return {"command": recognized_text, "text": "警报已解除", "alert": current_alert_status}
        else:
            # 警报仍然激活，但用户说了其他话 (不是解除指令)
            # 保持警报，返回提示信息
            # log_multimodal(user_id, input_modality, recognized_text, "警报未解除，请先解除。")
            # await update_common_commands(user_id, recognized_text, db) # 这种情况下是否记录常用指令？
            return {"command": recognized_text, "text": "警报未解除，请先解除警报或确认您的指令。", "alert": current_alert_status}

    # 3. 检查是否匹配预设指令
    response_text = get_command_response(processed_text)
    if response_text:
        try:
            tts_engine.say(response_text)
            tts_engine.runAndWait()
        except Exception as e:
            logger.warning(f"TTS 输出预设指令响应失败: {e}")
        
        log_multimodal(user_id, input_modality, recognized_text, response_text)
        await update_common_commands(user_id, recognized_text, db)
        return {"command": recognized_text, "text": response_text, "alert": current_alert_status}

    # 4. 如果未匹配任何预设指令，则调用大模型进行回答
    logger.info(f"用户 {user_id} (通过 {input_modality}) 发起大模型请求: '{recognized_text}'")
    try:
        memory = db.query(UserMemory).filter(UserMemory.user_id == user_id).one()
        history = json.loads(memory.content)
    except NoResultFound:
        history = []
        memory = UserMemory(user_id=user_id, content="[]") # 确保 content 是有效的JSON "[]"

    history.append({"role": "user", "content": processed_text}) # 使用处理过别名的文本

    MAX_HISTORY = 9
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    ai_reply = await call_zhipu_chat(history)
    history.append({"role": "assistant", "content": ai_reply})
    memory.content = json.dumps(history, ensure_ascii=False)
    db.merge(memory) # 使用 merge 来处理新建或更新
    db.commit()
    logger.info(f"用户 {user_id} 对话历史已更新")
    
    try:
        tts_engine.say(ai_reply)
        tts_engine.runAndWait()
    except Exception as e:
        logger.warning(f"TTS 输出大模型回复失败: {e}")

    log_multimodal(user_id, input_modality, recognized_text, ai_reply)
    await update_common_commands(user_id, recognized_text, db) # 大模型回复也算常用
    return {"command": recognized_text, "text": ai_reply, "alert": current_alert_status}


async def process_text_command(text_command: str, user_id: int, db: Session) -> Dict[str, Any]:
    """
    处理直接输入的文本指令并返回响应。
    """
    try:
        processed_text = text_command.strip()
        if not processed_text:
            logger.warning(f"用户 {user_id} 输入的文本指令为空。")
            return {"command": "", "text": "请输入有效指令。", "alert": alert_active}

        logger.info(f"处理文本指令: '{processed_text}' for user_id: {user_id}")
        # 调用核心文本处理逻辑
        return await _process_recognized_text(processed_text, user_id, db, input_modality="text_input")

    except Exception as e:
        logger.error(f"文本指令处理错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文本指令处理错误: {str(e)}"
        )
