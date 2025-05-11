import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss
from pynput.mouse import Listener as MouseListener, Button, Controller as MouseController
from pynput.keyboard import Key, Listener as KeyboardListener
import win32api
import win32con
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout,
                           QHBoxLayout, QSlider, QLineEdit, QCheckBox, QPushButton,
                           QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import sys
import time
from filterpy.kalman import KalmanFilter # 确保 filterpy 已安装
import pyttsx3
import threading
import ctypes
import json
import torch
from enum import Enum
import queue
from scipy.optimize import linear_sum_assignment # For tracker

# --- 常量和配置 ---
CONFIG_FILE = "ultimate_aim_config_cn_v1.3.json" # 版本更新
YOLO_MODEL_PATH = 'pubg1550.pt'
CAPTURE_W, CAPTURE_H = 320, 320

# --- 卡尔曼滤波器状态 ---
class KalmanState(Enum):
    INACTIVE = 0; ACQUIRING = 1; TRACKING = 2; LOST_TEMPORARY = 3

kalman_state_chinese_map = {
    KalmanState.INACTIVE: "未激活", KalmanState.ACQUIRING: "搜索目标",
    KalmanState.TRACKING: "追踪中", KalmanState.LOST_TEMPORARY: "短暂丢失"
}

# --- 全局配置和状态 ---
config = {}
class GlobalState:
    def __init__(self):
        self._lock = threading.Lock()
        self._is_ai_running = False
        self._is_aiming_active = False
        self._kalman_state_ui = KalmanState.INACTIVE
        self._last_known_target_id_ui = None
        self.backend_kalman_state = KalmanState.INACTIVE
        self.backend_current_aim_target_id = None
        self.backend_frames_since_aim_kf_seen = 0
        self.backend_mouse_dx_accumulator = 0.0
        self.backend_mouse_dy_accumulator = 0.0

    @property
    def is_ai_running(self):
        with self._lock: return self._is_ai_running
    @is_ai_running.setter
    def is_ai_running(self, value):
        with self._lock: self._is_ai_running = value

    @property
    def is_aiming_active(self):
        with self._lock: return self._is_aiming_active
    @is_aiming_active.setter
    def is_aiming_active(self, value):
        with self._lock: self._is_aiming_active = value

    def get_ui_kalman_state(self):
        with self._lock: return self._kalman_state_ui
    def set_ui_kalman_state(self, state):
        with self._lock: self._kalman_state_ui = state

    def get_ui_last_known_target_id(self):
        with self._lock: return self._last_known_target_id_ui
    def set_ui_last_known_target_id(self, target_id):
        with self._lock: self._last_known_target_id_ui = target_id

global_state = GlobalState()

raw_frame_queue = queue.Queue(maxsize=2)
detection_results_queue = queue.Queue(maxsize=2)
aim_target_info_queue = queue.Queue(maxsize=2)
ui_preview_queue = queue.Queue(maxsize=1)

TTS_LOCK = threading.Lock()
engine = pyttsx3.init()
try:
    voices = engine.getProperty('voices')
    if voices: pass
except Exception as e: print(f"TTS引擎初始化错误: {e}")

YOLO_MODEL = None
try:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在于 {DEVICE} 设备上加载YOLO模型 '{YOLO_MODEL_PATH}'...")
    YOLO_MODEL = YOLO(YOLO_MODEL_PATH).to(DEVICE)
    print(f"YOLO模型成功加载到 {DEVICE} 设备。")
except Exception as e:
    print(f"严重错误: 无法加载YOLO模型。程序退出。详情: {e}")
    sys.exit(1)

SCT_INSTANCE = mss()
if not SCT_INSTANCE.monitors: print("严重错误: MSS未能找到任何显示器。"); sys.exit(1)
PRIMARY_MONITOR = SCT_INSTANCE.monitors[1]

AIM_KF = KalmanFilter(dim_x=4, dim_z=2)
STOP_EVENT = threading.Event()

class UiUpdater(QObject):
    update_preview_signal = pyqtSignal(object)
    update_debug_text_signal = pyqtSignal(str)
ui_updater_instance = UiUpdater()

class AimAssistGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"终极AI辅助 v1.3 (鼠标控制优化) - {config.get('app_name', '')}") # 更新版本号
        self.mouse_controller = MouseController()
        self.status_label = QLabel("状态: 初始化中...")
        self.debug_label = QLabel("调试信息: 等待...")
        self.preview_label_widget = QLabel("AI预览画面")
        self.preview_label_widget.setFixedSize(CAPTURE_W, CAPTURE_H)
        self.preview_label_widget.setStyleSheet("background-color: #333; color: white; border: 1px solid #555;")
        self.preview_label_widget.setAlignment(Qt.AlignCenter)

        self._init_ui_controls()
        self.load_app_settings()

        ui_updater_instance.update_preview_signal.connect(self.update_preview_slot)
        ui_updater_instance.update_debug_text_signal.connect(self.debug_label.setText)

        self.mouse_listener = MouseListener(on_click=self._on_mouse_click)
        self.mouse_listener.start()
        self.keyboard_listener = KeyboardListener(on_press=self._on_key_press)
        self.keyboard_listener.start()

        self.ui_refresh_timer = QTimer(self)
        self.ui_refresh_timer.timeout.connect(self._refresh_ui_status)
        self.ui_refresh_timer.start(200)

    def _init_ui_controls(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.preview_label_widget)
        controls_group = QGroupBox("参数控制")
        controls_layout = QGridLayout()
        self.sliders_config = {
            'x_sens': ("X轴灵敏度:", "x_sensitivity", 0.5, 10, 200, 100, "{:.2f}"),
            'y_sens': ("Y轴灵敏度:", "y_sensitivity", 0.4, 10, 200, 100, "{:.2f}"),
            'conf': ("置信度阈值:", "confidence_threshold", 0.5, 10, 100, 100, "{:.2f}"),
            'recoil_y': ("Y轴后坐力补偿:", "recoil_compensation_y", 30, 0, 100, 1, "{}"),  # 默认值改为30
            'fov': ("瞄准范围半径(px):", "fov_radius_pixels", 80, 10, 160, 1, "{}"),
            'smoothing': ("平滑系数:", "smoothing_factor", 0.5, 0, 100, 100, "{:.2f}"),
            'max_move': ("每帧最大移动(px):", "max_mouse_move_per_frame", 20, 5, 100, 1, "{}"),
            'deadzone': ("死区像素:", "deadzone_pixels", 5, 0, 20, 1, "{}")
            # 移除预测强度滑块，使用简化的鼠标移动逻辑
        }
        self.slider_widgets = {}
        row = 0
        for name, (label_text, cfg_key, def_val, s_min, s_max, mult, fmt) in self.sliders_config.items():
            # config.get(cfg_key, def_val) 确保在 config 未完全加载时也有值
            current_cfg_val = config.get(cfg_key, def_val)
            slider_val = int(current_cfg_val * mult)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(s_min, s_max)
            slider.setValue(slider_val)
            text_edit = QLineEdit(fmt.format(current_cfg_val)) # 使用实际值格式化
            text_edit.setReadOnly(True)
            text_edit.setFixedWidth(60)

            controls_layout.addWidget(QLabel(label_text), row, 0)
            controls_layout.addWidget(slider, row, 1)
            controls_layout.addWidget(text_edit, row, 2)
            self.slider_widgets[name] = (slider, text_edit)
            row += 1
            slider.valueChanged.connect(
                lambda v, k=cfg_key, s=text_edit, m=mult, f=fmt:
                self._update_config_and_ui(k, v / m if m != 1 else v, s, f.format(v / m if m != 1 else v))
            )
        self.preview_checkbox = QCheckBox("显示AI预览画面")
        self.preview_checkbox.setChecked(config.get("show_preview", True))
        self.preview_checkbox.stateChanged.connect(
            lambda state: self._update_config_and_ui("show_preview", state == Qt.Checked)
        )
        controls_layout.addWidget(self.preview_checkbox, row, 0, 1, 3)
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.debug_label)
        settings_io_layout = QHBoxLayout()
        save_btn = QPushButton("保存设置"); load_btn = QPushButton("加载设置")
        save_btn.clicked.connect(self.save_app_settings)
        load_btn.clicked.connect(self.load_app_settings)
        settings_io_layout.addWidget(save_btn); settings_io_layout.addWidget(load_btn)
        main_layout.addLayout(settings_io_layout)
        self.setLayout(main_layout)

    def _update_config_and_ui(self, key, value, text_widget=None, formatted_text=None):
        config[key] = value
        if text_widget and formatted_text is not None: text_widget.setText(formatted_text)

    def _speak_async(self, text_to_speak):
        def speak_task(): global TTS_LOCK
        with TTS_LOCK:
            try: engine.say(text_to_speak); engine.runAndWait()
            except RuntimeError as e:
                if "run loop already started" in str(e).lower(): print(f"TTS信息: 尝试朗读 '{text_to_speak}' 时，上一个任务仍在进行。")
                else: print(f"TTS运行时错误: {e}")
            except Exception as e: print(f"TTS未知错误: {e}")
        threading.Thread(target=speak_task, daemon=True).start()

    def _on_mouse_click(self, x, y, button, pressed):
        global global_state, config
        current_time = time.time()
        button_name = getattr(button, 'name', None) or ( "left" if button == Button.left else str(button))
        toggle_key_name = config.get("aim_assist_toggle_key_name", "x1")

        # 优化G4键处理逻辑，减少处理时间
        if button_name == toggle_key_name and pressed:
            if current_time - getattr(self, '_last_toggle_time', 0) > 0.3:
                # 使用线程处理语音和状态更新，避免阻塞鼠标事件线程
                def toggle_ai_state():
                    global_state.is_ai_running = not global_state.is_ai_running
                    status_text = "AI辅助 已启动" if global_state.is_ai_running else "AI辅助 已关闭"
                    self._speak_async(status_text)
                    if not global_state.is_ai_running:
                        _reset_aim_kf_state(KalmanState.INACTIVE)

                threading.Thread(target=toggle_ai_state, daemon=True).start()
                setattr(self, '_last_toggle_time', current_time)
        elif button_name == "left":
            global_state.is_aiming_active = pressed
            if pressed:
                if global_state.is_ai_running: _reset_aim_kf_state(KalmanState.ACQUIRING)
            else:
                if global_state.backend_kalman_state != KalmanState.INACTIVE:
                    _reset_aim_kf_state(KalmanState.INACTIVE)

    def _on_key_press(self, key):
        try:
            if key == Key.f1: self._adjust_slider_value_by_name('recoil_y', 5, "后坐力补偿 {}")
            elif key == Key.f2: self._adjust_slider_value_by_name('recoil_y', -5, "后坐力补偿 {}")
            elif key == Key.f3: self._adjust_slider_value_by_name('x_sens', 5, "X轴灵敏度 {:.2f}")
            elif key == Key.f4: self._adjust_slider_value_by_name('x_sens', -5, "X轴灵敏度 {:.2f}")
        except Exception as e: print(f"快捷键处理错误: {e}")

    def _adjust_slider_value_by_name(self, slider_name, change_in_slider_units, speak_format):
        if slider_name in self.slider_widgets:
            slider, _ = self.slider_widgets[slider_name]
            _, _, _, _, _, mult, fmt_str = self.sliders_config[slider_name]
            new_val_slider = max(slider.minimum(), min(slider.maximum(), slider.value() + change_in_slider_units))
            slider.setValue(new_val_slider)
            actual_value_for_speaking = new_val_slider / mult if mult != 1 else new_val_slider
            try: self._speak_async(speak_format.format(actual_value_for_speaking))
            except Exception as e: print(f"语音播报格式错误: {e}")

    @pyqtSlot(object)
    def update_preview_slot(self, frame_bgr_np):
        if config.get("show_preview", True) and frame_bgr_np is not None and isinstance(frame_bgr_np, np.ndarray):
            try:
                h, w, ch = frame_bgr_np.shape
                bytes_per_line = ch * w
                rgb_frame = cv2.cvtColor(frame_bgr_np, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.preview_label_widget.setPixmap(QPixmap.fromImage(q_img))
            except Exception as e: print(f"UI预览槽函数错误: {e}, frame shape: {frame_bgr_np.shape if isinstance(frame_bgr_np, np.ndarray) else 'Not ndarray'}")
        elif not config.get("show_preview", True): self.preview_label_widget.clear()

    def _refresh_ui_status(self):
        ai_status = "运行中" if global_state.is_ai_running else "已停止"
        aim_status = "瞄准激活" if global_state.is_aiming_active else "瞄准未激活"
        self.status_label.setText(f"状态: AI {ai_status} | {aim_status}")

    def load_app_settings(self):
        load_configuration(CONFIG_FILE)
        for name, (label_text, cfg_key, def_val, s_min, s_max, mult, fmt) in self.sliders_config.items():
            if name in self.slider_widgets:
                slider, text_edit = self.slider_widgets[name]
                value_from_config = config.get(cfg_key, def_val)
                slider_val_to_set = int(float(value_from_config) * mult) # Ensure float before mult
                slider.setValue(slider_val_to_set)
                text_edit.setText(fmt.format(value_from_config))
        self.preview_checkbox.setChecked(config.get("show_preview", True))
        self._speak_async("设置已加载")
        print(f"设置已从 '{CONFIG_FILE}' 加载并应用到UI。")

    def save_app_settings(self):
        save_configuration(CONFIG_FILE)
        self._speak_async("设置已保存")

    def closeEvent(self, event):
        print("UI正在关闭，请求保存设置并停止线程...")
        self.save_app_settings()
        STOP_EVENT.set()
        if hasattr(self, 'keyboard_listener') and self.keyboard_listener.is_alive(): self.keyboard_listener.stop()
        if hasattr(self, 'mouse_listener') and self.mouse_listener.is_alive(): self.mouse_listener.stop()
        print("UI关闭事件处理完毕。")
        event.accept()

def get_float_config(key, default_val):
    val = config.get(key, default_val)
    try: return float(val)
    except (ValueError, TypeError): return float(default_val)
def get_int_config(key, default_val):
    val = config.get(key, default_val)
    try: return int(val)
    except (ValueError, TypeError): return int(default_val)

def load_configuration(path):
    global config, AIM_KF
    default_config = {
        "app_name": "终极AI辅助", "input_fps": 45, "capture_width": CAPTURE_W, "capture_height": CAPTURE_H,
        "yolo_model_path": YOLO_MODEL_PATH, "yolo_device": DEVICE,
        "yolo_conf_threshold": 0.45, "target_class_id": 0,
        "kalman_initial_uncertainty": 100.0, "kalman_measurement_noise_r": 20.0,
        "kalman_process_noise_q_pos": 0.05, "kalman_process_noise_q_vel": 0.005,
        "tracker_kalman_r_diag": 5.0, "tracker_kalman_q_pos_factor": 0.02,
        "tracker_kalman_q_vel_factor": 0.002,
        "tracker_max_age": 25, "tracker_min_hits": 3, "tracker_iou_threshold": 0.3,
        "x_sensitivity": 0.5, "y_sensitivity": 0.4, # 灵敏度设置
        "recoil_compensation_y": 30, "smoothing_factor": 0.5, # 平滑度设置，默认后坐力补偿值设为30
        "max_mouse_move_per_frame": 20, "fov_radius_pixels": 80,
        "show_preview": True, "aim_assist_toggle_key_name": "x1",
        "target_stickiness_factor": 0.7,
        "tracker_max_frames_lost_for_prediction": 5,
        "deadzone_pixels": 5,
        # 移除预测强度参数，使用简化的鼠标移动逻辑
    }
    try:
        with open(path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            config = {**default_config, **loaded_config}
        print(f"配置已从 '{path}' 加载。")
    except FileNotFoundError:
        print(f"配置文件 '{path}' 未找到，使用默认配置并保存。")
        config = default_config
        save_configuration(path)
    except Exception as e:
        print(f"加载配置错误: {e}，使用默认配置。")
        config = default_config

    config["kalman_measurement_noise_r"] = get_float_config("kalman_measurement_noise_r", default_config["kalman_measurement_noise_r"])
    config["kalman_process_noise_q_pos"] = get_float_config("kalman_process_noise_q_pos", default_config["kalman_process_noise_q_pos"])
    config["kalman_process_noise_q_vel"] = get_float_config("kalman_process_noise_q_vel", default_config["kalman_process_noise_q_vel"])
    config["kalman_initial_uncertainty"] = get_float_config("kalman_initial_uncertainty", default_config["kalman_initial_uncertainty"])

    AIM_KF.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    AIM_KF.H = np.array([[1,0,0,0],[0,1,0,0]])
    AIM_KF.R = np.eye(2) * config["kalman_measurement_noise_r"]
    q_pos = config["kalman_process_noise_q_pos"]
    q_vel = config["kalman_process_noise_q_vel"]
    AIM_KF.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
    AIM_KF.x = np.zeros((4,1))
    AIM_KF.P = np.eye(4) * config["kalman_initial_uncertainty"]

def save_configuration(path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e: print(f"保存配置错误: {e}")

def iou_batch(bb_test, bb_gt):
    if bb_test.ndim == 1: bb_test = bb_test.reshape(1, -1)
    if bb_gt.ndim == 1: bb_gt = bb_gt.reshape(1, -1)
    if bb_test.shape[0] == 0 or bb_gt.shape[0] == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]))

    iou_matrix = np.zeros((bb_test.shape[0], bb_gt.shape[0]))
    for i in range(bb_test.shape[0]):
        for j in range(bb_gt.shape[0]):
            b1 = bb_test[i,:4]; b2 = bb_gt[j,:4] # Ensure only first 4 elements are used
            # 确保使用标量值而不是数组
            b1_0, b1_1, b1_2, b1_3 = float(b1[0]), float(b1[1]), float(b1[2]), float(b1[3])
            b2_0, b2_1, b2_2, b2_3 = float(b2[0]), float(b2[1]), float(b2[2]), float(b2[3])

            xx1 = np.maximum(b1_0, b2_0); yy1 = np.maximum(b1_1, b2_1)
            xx2 = np.minimum(b1_2, b2_2); yy2 = np.minimum(b1_3, b2_3)
            w = np.maximum(0., xx2 - xx1); h = np.maximum(0., yy2 - yy1)
            wh = w * h
            area1 = (b1_2-b1_0)*(b1_3-b1_1); area2 = (b2_2-b2_0)*(b2_3-b2_1)
            if area1 <= 1e-5 or area2 <= 1e-5 : iou_matrix[i,j] = 0; continue # Prevent division by zero with tiny areas
            union = area1 + area2 - wh
            if union <= 1e-5 : iou_matrix[i,j] = 0; continue
            iou_matrix[i,j] = wh / union
    return iou_matrix

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox_xyxy_conf):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        r_diag = get_float_config("tracker_kalman_r_diag", 5.0)
        q_pos_f = get_float_config("tracker_kalman_q_pos_factor", 0.02)
        q_vel_f = get_float_config("tracker_kalman_q_vel_factor", 0.002)

        self.kf.R = np.eye(4) * r_diag
        self.kf.P = np.eye(7) * 10.
        self.kf.P[4:,4:] *= 100.

        self.kf.Q = np.eye(7) * 1e-5 # Base small noise
        self.kf.Q[0,0] = self.kf.Q[1,1] = q_pos_f**2
        self.kf.Q[2,2] = self.kf.Q[3,3] = (q_pos_f*0.1)**2
        self.kf.Q[4,4] = self.kf.Q[5,5] = q_vel_f**2
        self.kf.Q[6,6] = (q_vel_f*0.1)**2

        initial_z = self._to_z(bbox_xyxy_conf[:4])
        if np.any(np.isnan(initial_z)) or np.any(np.isinf(initial_z)):
            self.kf.x[:4] = self._to_z(np.array([CAPTURE_W/2-1,CAPTURE_H/2-1,CAPTURE_W/2+1,CAPTURE_H/2+1])).reshape((4,1)) # Center fallback
            print(f"警告: KalmanBoxTracker ID {self.id if hasattr(self, 'id') else 'New'} 初始化时bbox无效: {bbox_xyxy_conf[:4]}")
        else: self.kf.x[:4] = initial_z.reshape((4,1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count; KalmanBoxTracker.count += 1
        self.hits = 1; self.hit_streak = 1; self.age = 0
        self.last_observation_conf = bbox_xyxy_conf[4]

    def _to_z(self, bbox):
        w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
        if w <= 1e-3: w = 1e-3
        if h <= 1e-3: h = 1e-3
        return np.array([bbox[0]+w/2., bbox[1]+h/2., w*h, w/h])

    def _to_xyxy(self, x_state):
        cx, cy, s, r = x_state[0], x_state[1], x_state[2], x_state[3]
        if s <= 1e-3: s = 1e-3
        if r <= 1e-3: r = 1e-3
        h = np.sqrt(s/r); w = r*h
        if np.isnan(w) or np.isinf(w) or w <= 0: w = 1.0
        if np.isnan(h) or np.isinf(h) or h <= 0: h = 1.0
        return np.array([cx-w/2., cy-h/2., cx+w/2., cy+h/2.])

    def update(self, bbox_xyxy_conf):
        self.time_since_update = 0; self.hits += 1; self.hit_streak += 1
        measurement_z = self._to_z(bbox_xyxy_conf[:4])
        if np.any(np.isnan(measurement_z)) or np.any(np.isinf(measurement_z)):
            return
        self.kf.update(measurement_z.reshape((4,1)))
        if np.any(np.isnan(self.kf.x)) or np.any(np.isinf(self.kf.x)):
            self.kf.P = np.eye(self.kf.dim_x) * get_float_config("kalman_initial_uncertainty", 100.0) * 10
            self.kf.x[4:] = 0
        self.last_observation_conf = bbox_xyxy_conf[4]

    def predict(self):
        if (self.kf.x[6]+self.kf.x[2])<=1e-3 : self.kf.x[6] *= 0.0
        self.kf.predict()
        if np.any(np.isnan(self.kf.x)) or np.any(np.isinf(self.kf.x)): return np.array([-1,-1,-1,-1])
        self.age += 1
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1
        predicted_bbox = self._to_xyxy(self.kf.x)
        if np.any(np.isnan(predicted_bbox)) or np.any(np.isinf(predicted_bbox)): return np.array([-1,-1,-1,-1])
        return predicted_bbox

    def get_state(self):
        current_bbox = self._to_xyxy(self.kf.x)
        if np.any(np.isnan(current_bbox)) or np.any(np.isinf(current_bbox)): return np.array([-1,-1,-1,-1])
        return current_bbox

class InputModule(threading.Thread):
    def __init__(self):
        super().__init__(name="InputModule", daemon=True)
        self.sct = None # Initialize in run
        self.capture_rect = {}
    def run(self):
        print("输入模块已启动。")
        self.sct = mss() # Create mss instance in the thread
        self.capture_rect = {
            'left': PRIMARY_MONITOR['left'] + (PRIMARY_MONITOR['width'] - get_int_config("capture_width", CAPTURE_W)) // 2,
            'top': PRIMARY_MONITOR['top'] + (PRIMARY_MONITOR['height'] - get_int_config("capture_height", CAPTURE_H)) // 2,
            'width': get_int_config("capture_width", CAPTURE_W),
            'height': get_int_config("capture_height", CAPTURE_H)
        }
        target_frame_time = 1.0 / get_int_config("input_fps", 30)
        last_frame_time = time.perf_counter()
        while not STOP_EVENT.is_set():
            if not global_state.is_ai_running: time.sleep(0.1); continue
            current_time = time.perf_counter()
            sleep_duration = target_frame_time - (current_time - last_frame_time)
            if sleep_duration > 0: time.sleep(sleep_duration)
            last_frame_time = time.perf_counter() # Update after potential sleep
            try:
                img_bgra = np.array(self.sct.grab(self.capture_rect))
                raw_frame_queue.put({'timestamp': last_frame_time, 'frame_bgra': img_bgra}, timeout=target_frame_time*0.9)
            except queue.Full: pass
            except Exception as e:
                print(f"InputModule Error: {e}. Re-initializing sct.")
                try: self.sct.close()
                except: pass
                self.sct = mss() # Re-initialize on error
        if self.sct: self.sct.close() # Cleanup
        print("输入模块已停止。")

class DetectionModule(threading.Thread):
    def __init__(self):
        super().__init__(name="DetectionModule", daemon=True)
        self.model = YOLO_MODEL
    def run(self):
        print("检测模块已启动。")
        while not STOP_EVENT.is_set():
            if not global_state.is_ai_running: time.sleep(0.1); continue
            try:
                msg = raw_frame_queue.get(timeout=0.2) # Longer timeout for get
                frame_bgra = msg['frame_bgra']; timestamp = msg['timestamp']
                frame_rgb = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2RGB)
                results = self.model(frame_rgb, verbose=False,
                                      conf=get_float_config("yolo_conf_threshold",0.45)*0.85,
                                      iou=0.5, classes=[get_int_config("target_class_id",0)])
                detections_data = []
                if results and results[0].boxes:
                    for box_data in results[0].boxes:
                        conf = float(box_data.conf[0])
                        if conf >= get_float_config("confidence_threshold", 0.5):
                            xyxy = box_data.xyxy[0].cpu().numpy().astype(int)
                            detections_data.append(np.concatenate((xyxy, [conf])))
                detection_results_queue.put({'timestamp':timestamp, 'frame_bgra': frame_bgra, 'detections': detections_data}, timeout=0.1)
                raw_frame_queue.task_done()
            except queue.Empty: pass
            except Exception as e: print(f"DetectionModule Error: {e}")
        print("检测模块已停止。")

class TrackingFilterModule(threading.Thread):
    def __init__(self):
        super().__init__(name="TrackingFilterModule", daemon=True)
        self.trackers = []
        self.aim_kf = AIM_KF
        KalmanBoxTracker.count = 0
    def _associate(self, detections_xyxy_conf, trackers_predicted_bboxes_xyxy):
        if detections_xyxy_conf.ndim == 1 and detections_xyxy_conf.shape[0] > 0 : detections_xyxy_conf = detections_xyxy_conf.reshape(1,-1)
        if trackers_predicted_bboxes_xyxy.ndim == 1 and trackers_predicted_bboxes_xyxy.shape[0] > 0 : trackers_predicted_bboxes_xyxy = trackers_predicted_bboxes_xyxy.reshape(1,-1)

        if trackers_predicted_bboxes_xyxy.shape[0] == 0:
            return np.empty((0,2),dtype=int), np.arange(len(detections_xyxy_conf)), np.empty((0),dtype=int)
        if detections_xyxy_conf.shape[0] == 0:
            return np.empty((0,2),dtype=int), np.empty((0),dtype=int), np.arange(len(trackers_predicted_bboxes_xyxy))

        iou_mat = iou_batch(detections_xyxy_conf[:,:4], trackers_predicted_bboxes_xyxy[:,:4]) # Ensure only xyxy for IoU
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched_pairs = []
        unmatched_dets = list(range(len(detections_xyxy_conf)))
        unmatched_trks = list(range(len(trackers_predicted_bboxes_xyxy)))
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r,c] >= get_float_config("tracker_iou_threshold", 0.25):
                matched_pairs.append([r,c])
                if r in unmatched_dets: unmatched_dets.remove(r)
                if c in unmatched_trks: unmatched_trks.remove(c)
        return np.array(matched_pairs), np.array(unmatched_dets), np.array(unmatched_trks)

    def _select_best_aim_target(self, current_tracked_infos):
        if not current_tracked_infos: return None
        fov_cx = get_int_config("capture_width",CAPTURE_W)/2
        fov_cy = get_int_config("capture_height",CAPTURE_H)/2
        fov_r = get_int_config("fov_radius_pixels", 100)
        valid_in_fov = []

        # 优先选择靠近中心的目标
        for info in current_tracked_infos:
            b = info['box']
            cx = (b[0]+b[2])/2
            cy = (b[1]+b[3])/2

            # 计算到中心的距离
            dist = np.sqrt((cx-fov_cx)**2 + (cy-fov_cy)**2)

            # 给Y轴距离更高的权重，使系统更倾向于选择水平方向上的目标
            # 这有助于减少垂直方向的过度移动
            y_weight = 1.2  # Y轴权重增加20%
            weighted_dist = np.sqrt((cx-fov_cx)**2 + ((cy-fov_cy)*y_weight)**2)

            if dist <= fov_r:
                info['distance_to_fov'] = dist
                info['weighted_distance'] = weighted_dist
                valid_in_fov.append(info)

        if not valid_in_fov: return None

        # 优先考虑当前已锁定的目标（粘性）
        stickiness = get_float_config("target_stickiness_factor", 0.7)
        if global_state.backend_current_aim_target_id is not None and stickiness > 0:
            for t in valid_in_fov:
                if t['id'] == global_state.backend_current_aim_target_id: return t

        # 按加权距离排序，选择最近的目标
        valid_in_fov.sort(key=lambda x: x['weighted_distance'])
        return valid_in_fov[0]

    def run(self):
        print("追踪滤波模块已启动。")
        while not STOP_EVENT.is_set():
            if not global_state.is_ai_running:
                if global_state.backend_kalman_state != KalmanState.INACTIVE: _reset_aim_kf_state(KalmanState.INACTIVE)
                time.sleep(0.1); continue
            try:
                msg = detection_results_queue.get(timeout=0.2) # Increased timeout
                timestamp = msg['timestamp']; frame_bgra = msg['frame_bgra']
                detections_xyxy_conf = np.array(msg['detections']) if msg['detections'] else np.empty((0,5))

                predicted_boxes_from_trackers = []
                valid_tracker_indices = []
                for idx, trk in enumerate(self.trackers):
                    pred_box = trk.predict()
                    if not (np.any(np.isnan(pred_box)) or np.any(np.isinf(pred_box)) or np.all(pred_box < 0)):
                        predicted_boxes_from_trackers.append(pred_box)
                        valid_tracker_indices.append(idx)

                predicted_boxes_np = np.array(predicted_boxes_from_trackers) if predicted_boxes_from_trackers else np.empty((0,4))
                matched, unm_dets, unm_trks_relative = self._associate(detections_xyxy_conf, predicted_boxes_np)

                for m_pair in matched:
                    original_tracker_idx = valid_tracker_indices[m_pair[1]]
                    self.trackers[original_tracker_idx].update(detections_xyxy_conf[m_pair[0]])
                for i in unm_dets: self.trackers.append(KalmanBoxTracker(detections_xyxy_conf[i]))

                current_tracked_infos_for_ui = []
                active_trackers_next_loop = []
                for trk in self.trackers:
                    current_state_box = trk.get_state()
                    if np.any(np.isnan(current_state_box)) or np.any(np.isinf(current_state_box)) or np.all(current_state_box < 0): continue
                    if trk.time_since_update < get_int_config("tracker_max_age",20) and \
                       (trk.hits >= get_int_config("tracker_min_hits",2) or trk.age < get_int_config("tracker_min_hits",2)):
                        current_tracked_infos_for_ui.append({'id': trk.id, 'box': current_state_box, 'conf': trk.last_observation_conf})
                        active_trackers_next_loop.append(trk)
                    elif trk.time_since_update >= get_int_config("tracker_max_age",20): pass
                    else: active_trackers_next_loop.append(trk)
                self.trackers = active_trackers_next_loop

                best_target_for_aiming = None
                if global_state.is_aiming_active: best_target_for_aiming = self._select_best_aim_target(current_tracked_infos_for_ui)

                current_aim_kf_s = global_state.backend_kalman_state
                new_aim_kf_s = current_aim_kf_s

                if best_target_for_aiming:
                    target_center = ((best_target_for_aiming['box'][0]+best_target_for_aiming['box'][2])/2, (best_target_for_aiming['box'][1]+best_target_for_aiming['box'][3])/2)
                    measurement = np.array(target_center).reshape(2,1)
                    if np.any(np.isnan(measurement)) or np.any(np.isinf(measurement)):
                        if current_aim_kf_s == KalmanState.TRACKING: new_aim_kf_s = KalmanState.LOST_TEMPORARY
                    else:
                        if current_aim_kf_s==KalmanState.INACTIVE or current_aim_kf_s==KalmanState.ACQUIRING or \
                           (global_state.backend_current_aim_target_id != best_target_for_aiming['id'] and current_aim_kf_s != KalmanState.LOST_TEMPORARY):
                            _reset_aim_kf_state(KalmanState.TRACKING, target_center)
                            global_state.backend_current_aim_target_id = best_target_for_aiming['id']
                        self.aim_kf.predict()
                        if np.any(np.isnan(self.aim_kf.x)) or np.any(np.isinf(self.aim_kf.x)):
                            _reset_aim_kf_state(KalmanState.ACQUIRING if global_state.is_aiming_active else KalmanState.INACTIVE, measurement if not (np.any(np.isnan(measurement)) or np.any(np.isinf(measurement))) else None)
                        else:
                            self.aim_kf.update(measurement)
                            if np.any(np.isnan(self.aim_kf.x)) or np.any(np.isinf(self.aim_kf.x)):
                                _reset_aim_kf_state(KalmanState.ACQUIRING if global_state.is_aiming_active else KalmanState.INACTIVE, measurement if not (np.any(np.isnan(measurement)) or np.any(np.isinf(measurement))) else None)
                            else: new_aim_kf_s = KalmanState.TRACKING; global_state.backend_frames_since_aim_kf_seen = 0
                else:
                    if current_aim_kf_s == KalmanState.TRACKING: new_aim_kf_s = KalmanState.LOST_TEMPORARY; global_state.backend_frames_since_aim_kf_seen = 1
                    elif current_aim_kf_s == KalmanState.LOST_TEMPORARY:
                        global_state.backend_frames_since_aim_kf_seen += 1
                        max_lost_f = get_int_config("tracker_max_frames_lost_for_prediction", 8)
                        if global_state.backend_frames_since_aim_kf_seen > max_lost_f :
                            new_aim_kf_s = KalmanState.ACQUIRING if global_state.is_aiming_active else KalmanState.INACTIVE
                            if new_aim_kf_s == KalmanState.INACTIVE : _reset_aim_kf_state(KalmanState.INACTIVE)
                        else:
                            self.aim_kf.predict()
                            if np.any(np.isnan(self.aim_kf.x)) or np.any(np.isinf(self.aim_kf.x)):
                                new_aim_kf_s = KalmanState.ACQUIRING if global_state.is_aiming_active else KalmanState.INACTIVE
                                if new_aim_kf_s == KalmanState.INACTIVE : _reset_aim_kf_state(KalmanState.INACTIVE)
                    elif global_state.is_aiming_active and current_aim_kf_s == KalmanState.INACTIVE: new_aim_kf_s = KalmanState.ACQUIRING
                global_state.backend_kalman_state = new_aim_kf_s

                aim_target_payload = None
                if global_state.backend_kalman_state in [KalmanState.TRACKING, KalmanState.LOST_TEMPORARY] and np.all(np.isfinite(self.aim_kf.x)):
                    pred_aim_x = self.aim_kf.x[0,0] + self.aim_kf.x[2,0]; pred_aim_y = self.aim_kf.x[1,0] + self.aim_kf.x[3,0]

                    # 添加安全检查，防止预测位置过远
                    if not (-CAPTURE_W*2 < pred_aim_x < CAPTURE_W*2 and -CAPTURE_H*2 < pred_aim_y < CAPTURE_H*2):
                        _reset_aim_kf_state(KalmanState.ACQUIRING if global_state.is_aiming_active else KalmanState.INACTIVE)
                        print("预测位置超出合理范围，重置卡尔曼滤波器")
                    else:
                        # 简化版本的速度限制 - 类似飞易来版本
                        # 限制X轴和Y轴速度，防止过快移动导致的跟踪不稳定
                        max_velocity = 20.0  # 最大允许速度

                        # 限制X轴速度
                        if abs(self.aim_kf.x[2,0]) > max_velocity:
                            self.aim_kf.x[2,0] = max_velocity if self.aim_kf.x[2,0] > 0 else -max_velocity

                        # 限制Y轴速度
                        if abs(self.aim_kf.x[3,0]) > max_velocity:
                            self.aim_kf.x[3,0] = max_velocity if self.aim_kf.x[3,0] > 0 else -max_velocity

                        # 使用当前位置而不是预测位置，减少预测导致的过度移动
                        current_x = self.aim_kf.x[0,0]
                        current_y = self.aim_kf.x[1,0]

                        aim_target_payload = {'predicted_center':(current_x, current_y), 'current_best_target_info':best_target_for_aiming}
                        try: aim_target_info_queue.put(aim_target_payload, timeout=0.01)
                        except queue.Full: pass

                display_frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                if config.get("show_preview", True):
                    for trk_info in current_tracked_infos_for_ui:
                        b=trk_info['box']; tid=trk_info['id']
                        # 确保使用标量值而不是数组
                        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                        cv2.rectangle(display_frame_bgr,(int(x1),int(y1)),(int(x2),int(y2)),(0,160,0),1)
                        cv2.putText(display_frame_bgr,str(tid),(int(x1),int(y1)-3),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,160,0),1)
                    if aim_target_payload and aim_target_payload.get('predicted_center'):
                        px,py = int(aim_target_payload['predicted_center'][0]), int(aim_target_payload['predicted_center'][1])
                        if 0<=px<CAPTURE_W and 0<=py<CAPTURE_H:
                            # 绘制目标中心点
                            cv2.circle(display_frame_bgr,(px,py),4,(255,0,255),-1)
                            # 绘制十字准星
                            cv2.line(display_frame_bgr, (px-10, py), (px+10, py), (255, 0, 255), 1)
                            cv2.line(display_frame_bgr, (px, py-10), (px, py+10), (255, 0, 255), 1)

                        target_info_box = aim_target_payload.get('current_best_target_info')
                        if target_info_box and target_info_box.get('box') is not None:
                            b_aim = target_info_box['box']
                            # 确保使用标量值而不是数组
                            x1, y1, x2, y2 = float(b_aim[0]), float(b_aim[1]), float(b_aim[2]), float(b_aim[3])
                            # 使用更粗的线条绘制目标框
                            cv2.rectangle(display_frame_bgr,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
                    cv2.circle(display_frame_bgr,(CAPTURE_W//2,CAPTURE_H//2),get_int_config("fov_radius_pixels",100),(255,255,0),1)
                    try: ui_preview_queue.put(display_frame_bgr, timeout=0.01)
                    except queue.Full: pass

                global_state.set_ui_kalman_state(global_state.backend_kalman_state)
                global_state.set_ui_last_known_target_id(global_state.backend_current_aim_target_id)
                detection_results_queue.task_done()
            except queue.Empty: time.sleep(0.005)
            except Exception as e: print(f"TrackingFilterModule Error: {e}")
        print("追踪滤波模块已停止。")

class ControlModule(threading.Thread):
    def __init__(self):
        super().__init__(name="ControlModule", daemon=True)
        self.mouse_driver = MSDKMouseOutput()
        self.frame_center_x = CAPTURE_W / 2.0
        self.frame_center_y = CAPTURE_H / 2.0
        self._last_move_time = 0
        # 添加目标位置和时间跟踪
        self.last_target_pos = None
        self.last_update_time = time.perf_counter()

    def run(self):
        print("控制模块已启动。")
        target_control_loop_time = 0.005  # 200Hz控制循环
        last_loop_time = time.perf_counter()

        while not STOP_EVENT.is_set():
            if not (global_state.is_ai_running and global_state.is_aiming_active):
                # 平滑过渡到0
                smoothing = get_float_config("smoothing_factor", 0.3)
                global_state.backend_mouse_dx_accumulator *= smoothing
                global_state.backend_mouse_dy_accumulator *= smoothing
                if abs(global_state.backend_mouse_dx_accumulator) < 0.1:
                    global_state.backend_mouse_dx_accumulator = 0.0
                if abs(global_state.backend_mouse_dy_accumulator) < 0.1:
                    global_state.backend_mouse_dy_accumulator = 0.0
                # 重置目标位置跟踪
                self.last_target_pos = None
                time.sleep(0.01)
                continue

            current_time = time.perf_counter()
            if current_time - self._last_move_time < 0.005:
                time.sleep(0.001)
                continue

            sleep_dur = target_control_loop_time - (current_time - last_loop_time)
            if sleep_dur > 0:
                time.sleep(sleep_dur * 0.9)
            last_loop_time = time.perf_counter()

            try:
                target_info = aim_target_info_queue.get(timeout=0.01)
                aim_target_info_queue.task_done()

                pred_x_f, pred_y_f = target_info['predicted_center']
                if not (np.isfinite(pred_x_f) and np.isfinite(pred_y_f)):
                    global_state.backend_mouse_dx_accumulator = 0.0
                    global_state.backend_mouse_dy_accumulator = 0.0
                    self.last_target_pos = None
                    continue

                # 计算目标与屏幕中心的差值(偏移量)
                raw_dx = pred_x_f - self.frame_center_x
                raw_dy = pred_y_f - self.frame_center_y

                # 增加死区，防止微小移动
                deadzone = get_int_config("deadzone_pixels", 5)
                if abs(raw_dx) < deadzone: raw_dx = 0
                if abs(raw_dy) < deadzone: raw_dy = 0

                # 获取灵敏度设置
                x_sensitivity = get_float_config("x_sensitivity", 0.5)
                y_sensitivity = get_float_config("y_sensitivity", 0.4)

                # 计算移动量
                dx = int(raw_dx * x_sensitivity)
                dy = int(raw_dy * y_sensitivity)

                # 添加后坐力补偿
                recoil_comp = get_int_config("recoil_compensation_y", 0)
                if recoil_comp > 0:
                    dy += recoil_comp

                # 应用固定平滑系数 - 简化为固定值，类似飞易来版本
                smoothing_factor = get_float_config("smoothing_factor", 0.5)
                move_x = int(dx * (1 - smoothing_factor))
                move_y = int(dy * (1 - smoothing_factor))

                # 确保移动量不会过小
                if abs(dx) > 5 and abs(move_x) < 1:
                    move_x = 1 if dx > 0 else -1
                if abs(dy) > 5 and abs(move_y) < 1:
                    move_y = 1 if dy > 0 else -1

                # 添加移动限制，防止单次移动过大
                max_move = get_int_config("max_mouse_move_per_frame", 20)
                if abs(move_x) > max_move:
                    move_x = max_move if move_x > 0 else -max_move
                if abs(move_y) > max_move:
                    move_y = max_move if move_y > 0 else -max_move

                # 移动鼠标
                if (abs(move_x) > 0 or abs(move_y) > 0) and self.mouse_driver:
                    self.mouse_driver.move_mouse(move_x, move_y)
                    self._last_move_time = time.perf_counter()

            except queue.Empty:
                # 平滑衰减
                global_state.backend_mouse_dx_accumulator *= 0.8
                global_state.backend_mouse_dy_accumulator *= 0.8
                if abs(global_state.backend_mouse_dx_accumulator) < 0.1:
                    global_state.backend_mouse_dx_accumulator = 0
                if abs(global_state.backend_mouse_dy_accumulator) < 0.1:
                    global_state.backend_mouse_dy_accumulator = 0
            except Exception as e:
                print(f"ControlModule Error: {e}")
        if self.mouse_driver:
            self.mouse_driver.close()
        print("控制模块已停止。")

class MSDKMouseOutput:
    def __init__(self):
        self.hdl = None
        self.dll = None
        self.is_init_success = False
        self._lock = threading.Lock()  # 添加锁以防止并发访问
        try:
            self.dll = ctypes.windll.LoadLibrary('./msdk.dll')
            self.dll.M_Open.restype = ctypes.c_uint64
            self.hdl = self.dll.M_Open(1)
            if self.hdl == 0 or self.hdl == ctypes.c_uint64(-1).value:
                self.hdl = self.dll.M_Open(0)
            if self.hdl != 0 and self.hdl != ctypes.c_uint64(-1).value:
                sw, sh = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
                if self.dll.M_ResolutionUsed(self.hdl, sw, sh) == 0:
                    self.is_init_success = True
                    print(f"MSDK驱动: 初始化成功，分辨率 {sw}x{sh}。")
                else:
                    self.dll.M_Close(self.hdl)
                    self.hdl = None
                    print("MSDK驱动: 设置分辨率失败。")
            else:
                self.hdl = None
                print(f"MSDK驱动: 打开设备失败。错误: {ctypes.get_last_error()}")
        except Exception as e:
            print(f"MSDK驱动: 初始化错误: {e}")

    def move_mouse(self, dx, dy):
        if not self.is_init_success:
            return

        with self._lock:  # 使用锁确保线程安全
            try:
                # 确保参数是整数并且在合理范围内
                dx_int = int(np.clip(dx, -100, 100))
                dy_int = int(np.clip(dy, -100, 100))

                # 简化的鼠标移动逻辑，类似飞易来版本
                res = self.dll.M_MoveR2(self.hdl, dx_int, dy_int)
                if res != 0:
                    print(f"MSDK M_MoveR2 失败，代码: {res}。")
                    # 如果移动失败，可以考虑重置瞄准状态
                    if global_state.backend_kalman_state != KalmanState.INACTIVE:
                        _reset_aim_kf_state(KalmanState.INACTIVE)
            except TypeError as te:
                print(f"MSDK_MoveR2类型错误: {te} (dx:{dx}, dy:{dy})")
            except Exception as e:
                print(f"MSDK_MoveR2错误: {e}")
    def close(self):
        if self.hdl and self.dll and self.is_init_success:
            try: self.dll.M_Close(self.hdl); print("MSDK驱动: 已关闭。")
            except Exception as e: print(f"MSDK驱动关闭错误: {e}")
        self.hdl=None; self.dll=None; self.is_init_success=False

def _reset_aim_kf_state(new_kf_state: KalmanState, initial_measurement=None):
    global global_state, AIM_KF, config
    global_state.backend_kalman_state = new_kf_state
    AIM_KF.x = np.zeros((4, 1))
    if new_kf_state == KalmanState.ACQUIRING and initial_measurement is not None and \
       len(initial_measurement)==2 and np.all(np.isfinite(initial_measurement)):
        AIM_KF.x[0,0]=initial_measurement[0]; AIM_KF.x[1,0]=initial_measurement[1]
        AIM_KF.x[2:,0]=0
    AIM_KF.P = np.eye(4) * get_float_config("kalman_initial_uncertainty", 100.0)
    global_state.backend_current_aim_target_id = None
    global_state.backend_frames_since_aim_kf_seen = 0

def _cleanup_resources():
    global TTS_LOCK
    print("正在清理主资源...")
    with TTS_LOCK:
        try:
            if hasattr(engine, 'isBusy') and engine.isBusy(): engine.stop()
        except Exception as e: print(f"尝试停止TTS引擎时出错: {e}")
    print("TTS引擎已尝试停止。")

GUI_APP_INSTANCE = None
# 修改主程序退出逻辑，确保鼠标驱动最后关闭
if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    load_configuration(CONFIG_FILE)
    input_thread = InputModule()
    detection_thread = DetectionModule()
    tracking_thread = TrackingFilterModule()
    control_thread = ControlModule()
    threads = [input_thread, detection_thread, tracking_thread, control_thread]
    for t in threads: t.start()
    GUI_APP_INSTANCE = AimAssistGUI()
    GUI_APP_INSTANCE.show()
    ui_update_display_timer = QTimer()
    def process_ui_queue():
        try:
            frame_bgr = ui_preview_queue.get_nowait()
            ui_updater_instance.update_preview_signal.emit(frame_bgr)
            ui_preview_queue.task_done()
        except queue.Empty: pass
        except Exception as e: print(f"UI队列处理错误: {e}")
        kf_s = global_state.get_ui_kalman_state()
        tid = global_state.get_ui_last_known_target_id()
        debug_text = f"调试: KF {kalman_state_chinese_map.get(kf_s, kf_s.name)} | ID: {str(tid if tid is not None else '无')}"
        ui_updater_instance.update_debug_text_signal.emit(debug_text)
    ui_update_display_timer.timeout.connect(process_ui_queue)
    ui_update_display_timer.start(33) # ~30FPS UI update for preview
    exit_code = app.exec_()
    print("UI已关闭，正在停止后端线程...")
    STOP_EVENT.set()
    for t in threads:
        if t.is_alive():
            print(f"正在等待 {t.name} 退出...")
            t.join(timeout=2.0) # Shorter timeout
            if t.is_alive(): print(f"警告: 线程 {t.name} 未能在超时内退出。")
    _cleanup_resources()
    print(f"应用程序退出，代码: {exit_code}。")
    sys.exit(exit_code)
