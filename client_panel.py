#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client Panel - Giao diện cho Client (chỉ xem)
Chỉ có quyền xem Live View và crops, không được chỉnh sửa
"""

import sys
import logging as log
from pathlib import Path
from time import perf_counter, time
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea,
    QStatusBar, QSplitter, QToolBar, QAction, QSizePolicy,
    QMenu, QFileDialog, QDialog, QDialogButtonBox,
    QFormLayout, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtWidgets import QApplication

try:
    from openvino import Core, get_version
except ImportError:
    print("Vui lòng cài đặt OpenVINO: pip install openvino")
    sys.exit(1)

try:
    from utils import (
        crop,
        FaceDetector,
        LandmarksDetector,
        FaceIdentifier,
        FacesDatabase
    )
except ImportError:
    print("Không thể import utils. Vui lòng đảm bảo file utils.py tồn tại.")
    sys.exit(1)

from database import Database
from crops_manager import CropsManager
from events_manager import EventsManager
from models import CustomerSegment, EventType


class VideoThread(QThread):
    """Thread xử lý video để không block UI"""

    change_pixmap_signal = pyqtSignal(np.ndarray)  # Frame hiển thị
    face_detected_signal = pyqtSignal(list)  # List các faces detected
    fps_signal = pyqtSignal(float)  # FPS
    error_signal = pyqtSignal(str)  # Error message

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = None

        # Models
        self.face_detector = None
        self.landmarks_detector = None
        self.face_identifier = None
        self.faces_database = None

        # Source
        self.source = 0  # Default webcam
        self.frame_count = 0
        self.start_time = perf_counter()

        # Tracking cho mỗi người - để tránh spam
        self.face_tracking = {}  # {face_id: {'last_capture_time': timestamp, 'count': int}}
        self.capture_interval = 2.0  # Default, sẽ được cập nhật từ database
        self.db = None  # Sẽ được set từ ClientPanel
        self.events_manager = None  # Sẽ được set từ ClientPanel
        self.crops_manager = None  # Sẽ được set từ ClientPanel
        self.camera_id = None  # Sẽ được set từ ClientPanel

    def set_models(self, face_detector, landmarks_detector, face_identifier, faces_database):
        """Set models đã load"""
        self.face_detector = face_detector
        self.landmarks_detector = landmarks_detector
        self.face_identifier = face_identifier
        self.faces_database = faces_database

    def set_database(self, db):
        """Set database để load capture_interval"""
        self.db = db
        if self.db:
            self.capture_interval = self.db.get_capture_interval()

    def set_managers(self, events_manager, crops_manager, camera_id=None):
        """Set managers để lưu events và crops"""
        self.events_manager = events_manager
        self.crops_manager = crops_manager
        self.camera_id = camera_id

    def set_source(self, source):
        """Set video source"""
        self.source = source

    def run(self):
        """Main loop"""
        # Mở camera/video
        try:
            if isinstance(self.source, str) and not self.source.isdigit():
                self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(int(self.source))

            if not self.cap.isOpened():
                self.error_signal.emit(f"Không thể mở nguồn video: {self.source}")
                return

        except Exception as e:
            self.error_signal.emit(f"Lỗi khi mở video: {e}")
            return

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame
            try:
                faces_info = []
                current_time = time()

                if self.face_detector and self.landmarks_detector and self.face_identifier:
                    # Detect faces
                    rois = self.face_detector.infer((frame,))

                    if len(rois) > 0:
                        # Detect landmarks
                        landmarks = self.landmarks_detector.infer((frame, rois))

                        # Identify faces
                        identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))

                        # Track detected face IDs in this frame
                        detected_face_ids = set()

                        # Draw results on frame
                        for roi, lm, identity in zip(rois, landmarks, identities):
                            # Get info
                            if identity.id != FaceIdentifier.UNKNOWN_ID:
                                label = self.face_identifier.get_identity_label(identity.id)
                                confidence = 100.0 * (1 - identity.distance)
                                text = f"{label} ({confidence:.1f}%)"
                                color = (0, 255, 0)  # Green
                                face_id = label
                            else:
                                label = "Unknown"
                                confidence = 0
                                text = "Unknown"
                                color = (0, 0, 255)  # Red
                                face_id = f"Unknown_{int(current_time)}"  # Unique ID cho unknown

                            detected_face_ids.add(face_id)

                            # Draw bounding box
                            xmin = max(int(roi.position[0]), 0)
                            ymin = max(int(roi.position[1]), 0)
                            xmax = min(int(roi.position[0] + roi.size[0]), frame.shape[1])
                            ymax = min(int(roi.position[1] + roi.size[1]), frame.shape[0])

                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                            # Draw landmarks
                            for point in lm:
                                x = int(xmin + roi.size[0] * point[0])
                                y = int(ymin + roi.size[1] * point[1])
                                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                            # Draw text
                            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (xmin, ymin - textsize[1] - 10),
                                        (xmin + textsize[0], ymin), color, -1)
                            cv2.putText(frame, text, (xmin, ymin - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                            # Reload capture_interval từ database (nếu có thay đổi)
                            if self.db:
                                self.capture_interval = self.db.get_capture_interval()
                            
                            # Kiểm tra xem có nên chụp ảnh không (theo interval từ database)
                            should_capture = False
                            if face_id not in self.face_tracking:
                                # Lần đầu tiên gặp người này -> chụp ngay
                                should_capture = True
                                self.face_tracking[face_id] = {
                                    'last_capture_time': current_time,
                                    'count': 1
                                }
                            else:
                                # Kiểm tra đã qua capture_interval chưa
                                time_since_last = current_time - self.face_tracking[face_id]['last_capture_time']
                                if time_since_last >= self.capture_interval:
                                    should_capture = True
                                    self.face_tracking[face_id]['last_capture_time'] = current_time
                                    self.face_tracking[face_id]['count'] += 1

                            # Chỉ emit face info nếu cần chụp (theo capture_interval)
                            if should_capture:
                                # Crop face for display
                                face_crop = crop(frame, roi)

                                # Lưu events và crops vào database
                                event_id = None
                                crop_id = None
                                customer_id = None
                                is_known = (identity.id != FaceIdentifier.UNKNOWN_ID)
                                is_new_session = False
                                event_info = None
                                unknown_event_info = None
                                
                                if is_known:
                                    # Khách quen - lưu events và crops (session-based tracking)
                                    if self.events_manager and self.crops_manager and self.db:
                                        try:
                                            # Get customer by face_id
                                            customer = self.db.get_customer_by_face_id(face_id)
                                            if customer:
                                                customer_id = customer.id
                                            else:
                                                # Create new customer if not exists
                                                customer_id = self.db.add_customer(
                                                    face_id=face_id,
                                                    name=label,
                                                    segment=CustomerSegment.NEW
                                                )
                                            
                                            bbox_tuple = (xmin, ymin, xmax, ymax)
                                            
                                            # Call events manager to handle recognition (session-based tracking)
                                            # on_face_recognized() sẽ trả về event_info cho TẤT CẢ customers
                                            # Chỉ tạo event khi có session mới, không phải mỗi frame
                                            event_info = self.events_manager.on_face_recognized(
                                                customer_name=label,
                                                customer_id=customer_id,
                                                confidence=confidence,
                                                bbox=bbox_tuple,
                                                camera_id=self.camera_id if self.camera_id else 0,
                                                face_id=face_id
                                            )
                                            
                                            if event_info:
                                                event_id = event_info.get('event_id')
                                                should_save_crop = event_info.get('should_save_crop', False)
                                                is_new_session = event_info.get('is_new_session', False)
                                                
                                                # Lưu crop nếu cần (cho session mới hoặc theo cooldown)
                                                if should_save_crop and event_id:
                                                    crop_id = self.crops_manager.save_crop(
                                                        face_image=face_crop,
                                                        customer_name=label,
                                                        customer_id=customer_id,
                                                        event_id=event_id,
                                                        bbox=bbox_tuple,
                                                        confidence=confidence
                                                    )
                                        except Exception as e:
                                            log.error(f"Error saving event/crop: {e}")
                                else:
                                    # Khách mới (Unknown) - session-based tracking
                                    if self.events_manager and self.crops_manager:
                                        try:
                                            bbox_tuple = (xmin, ymin, xmax, ymax)
                                            
                                            # Call events manager for unknown face (session-based)
                                            unknown_event_info = self.events_manager.on_unknown_face(
                                                confidence=confidence,
                                                bbox=bbox_tuple,
                                                camera_id=self.camera_id if self.camera_id else 0
                                            )
                                            
                                            # Lưu crop cho unknown face
                                            # Chiến lược:
                                            # - Chụp 1 ảnh khi xuất hiện lần đầu (session mới)
                                            # - Chụp thêm ảnh khi có thay đổi đủ lớn (di chuyển hoặc thời gian)
                                            if unknown_event_info:
                                                event_id = unknown_event_info.get('event_id')
                                                is_new_unknown_session = unknown_event_info.get('is_new_session', False)
                                                should_save_crop = unknown_event_info.get('should_save_crop', False)
                                                
                                                # Lưu crop nếu:
                                                # 1. Session mới (lần đầu xuất hiện)
                                                # 2. Hoặc should_save_crop = True (có thay đổi đủ lớn trong session đang tiếp diễn)
                                                if event_id and (is_new_unknown_session or should_save_crop):
                                                    crop_id = self.crops_manager.save_crop(
                                                        face_image=face_crop,
                                                        customer_name="Unknown",
                                                        customer_id=None,
                                                        event_id=event_id,
                                                        bbox=bbox_tuple,
                                                        confidence=confidence
                                                    )
                                                    # Set flag để hiển thị trong UI chỉ khi session mới
                                                    if is_new_unknown_session:
                                                        is_new_session = is_new_unknown_session
                                        except Exception as e:
                                            log.error(f"Error saving unknown crop: {e}")

                                # Chỉ emit face info để hiển thị nếu là session mới
                                # (tránh spam UI với cùng một người)
                                if is_new_session:
                                    # Session mới - hiển thị trong UI (cả known và unknown)
                                    faces_info.append({
                                        'crop': face_crop,
                                        'label': label,
                                        'confidence': confidence,
                                        'bbox': (xmin, ymin, xmax, ymax),
                                        'face_id': face_id,
                                        'timestamp': datetime.now().strftime("%H:%M:%S")
                                    })

                        # Cleanup tracking - xóa những người không còn xuất hiện trong frame
                        # (sau 10 giây không thấy thì xóa khỏi tracking)
                        faces_to_remove = []
                        for tracked_id in self.face_tracking:
                            if tracked_id not in detected_face_ids:
                                time_since_last = current_time - self.face_tracking[tracked_id]['last_capture_time']
                                if time_since_last > 10.0:
                                    faces_to_remove.append(tracked_id)
                        
                        for face_id in faces_to_remove:
                            del self.face_tracking[face_id]
                            
                        # Check timeout sessions định kỳ (mỗi 5 giây)
                        if self.frame_count % 150 == 0 and self.events_manager:  # ~5 seconds at 30fps
                            try:
                                self.events_manager.check_timeout_sessions()
                            except Exception as e:
                                log.error(f"Error checking timeout sessions: {e}")

                # Calculate FPS
                self.frame_count += 1
                elapsed = perf_counter() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0

                # Draw FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Emit signals
                self.change_pixmap_signal.emit(frame)
                self.fps_signal.emit(fps)

                # Chỉ emit face_detected_signal nếu có faces mới cần hiển thị
                if len(faces_info) > 0:
                    self.face_detected_signal.emit(faces_info)

            except Exception as e:
                log.error(f"Error processing frame: {e}")

        # Cleanup
        if self.cap:
            self.cap.release()

    def stop(self):
        """Stop thread"""
        self._run_flag = False
        self.face_tracking.clear()
        self.wait()


class ClientPanel(QMainWindow):
    """Client Panel - Chỉ xem, không chỉnh sửa"""

    # Signal for logout
    logout_signal = pyqtSignal()

    def __init__(self, user_info: dict, db: Database, parent=None):
        super().__init__(parent)
        self.user_info = user_info
        self.db = db
        self.setWindowTitle(f"Client Panel - {user_info['username']}")
        self.setGeometry(100, 100, 1400, 900)

        # Models
        self.core = None
        self.face_detector = None
        self.landmarks_detector = None
        self.face_identifier = None
        self.faces_database = None

        # Model paths - Load from database (same as Admin)
        model_config = self.db.get_model_config()
        self.model_fd_path = model_config['model_fd_path']
        self.model_lm_path = model_config['model_lm_path']
        self.model_reid_path = model_config['model_reid_path']
        self.gallery_path = model_config['gallery_path']

        # Video source configuration
        self.video_source = None  # Will be set via File menu
        self.video_source_type = None  # 'webcam', 'local', 'ip', 'rtsp'
        self.video_source_name = None  # Display name
        self.video_source_configured = False  # Flag to check if source is configured

        # Video thread
        self.video_thread = None

        # Managers for saving events and crops
        self.crops_manager = CropsManager(self.db, "./crops")
        self.events_manager = EventsManager(self.db)
        
        # Camera ID - default to None (will be set if using camera)
        self.camera_id = None

        # Displayed crops storage - để hiển thị crops theo thứ tự
        self.displayed_crops = []  # List of {face_id, crop_image, timestamp, is_known}
        self.max_displayed_crops = 20  # Giới hạn số lượng crops hiển thị

        # Model config refresh timer (check if Admin updated models)
        self.model_config_timer = QTimer()
        self.model_config_timer.timeout.connect(self.check_model_config_update)
        self.model_config_timer.setInterval(5000)  # Check every 5 seconds
        self.last_model_config = None  # Track last known config

        # UI
        self.init_ui()

        # Status
        self.models_loaded = False

        # Auto-load models on startup
        self.load_models_auto()
        
        # Start model config check timer
        self.model_config_timer.start()

    def init_ui(self):
        """Initialize UI"""
        # Set main window background - màu xám
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2d2d2d);
            }
            QGroupBox {
                border: none;
                margin-top: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #e0e0e0;
                background: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QSplitter {
                background: transparent;
            }
            QSplitter::handle {
                background: transparent;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        central_widget.setLayout(main_layout)

        # Toolbar
        self.create_toolbar()

        # Content area (split view)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Video display (70%)
        left_panel = self.create_video_panel()
        splitter.addWidget(left_panel)

        # Right panel - Face info (30%)
        right_panel = self.create_face_info_panel()
        splitter.addWidget(right_panel)

        # Set split ratio 70:30
        splitter.setSizes([980, 420])

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: rgba(50, 50, 50, 0.9);
                color: #e0e0e0;
                border: none;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Chế độ Client - Người dùng: {self.user_info['username']}")

    def create_toolbar(self):
        """Create toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(32, 32))
        toolbar.setStyleSheet("""
            QToolBar {
                background: rgba(50, 50, 50, 0.95);
                border: none;
                spacing: 5px;
            }
            QToolBar::separator {
                background: rgba(200, 200, 200, 0.3);
                width: 1px;
                margin: 5px;
            }
        """)
        self.addToolBar(toolbar)

        # User info
        user_label = QLabel(f"👤 {self.user_info['username']} (Client)")
        user_label.setStyleSheet("padding: 5px; font-weight: bold; color: #ffffff;")
        toolbar.addWidget(user_label)

        toolbar.addSeparator()

        # File menu for video source selection
        file_menu = QMenu("📁 File", self)
        
        # Webcam option
        webcam_action = QAction("📷 Webcam", self)
        webcam_action.setStatusTip("Sử dụng webcam (camera 0)")
        webcam_action.triggered.connect(self.configure_webcam)
        file_menu.addAction(webcam_action)
        
        # Video Local option
        local_video_action = QAction("🎬 Video Local", self)
        local_video_action.setStatusTip("Chọn video file từ máy tính")
        local_video_action.triggered.connect(self.configure_local_video)
        file_menu.addAction(local_video_action)
        
        # IP Camera option
        ip_camera_action = QAction("🌐 IP Camera", self)
        ip_camera_action.setStatusTip("Kết nối IP Camera")
        ip_camera_action.triggered.connect(self.configure_ip_camera)
        file_menu.addAction(ip_camera_action)
        
        # RTSP Camera option
        rtsp_camera_action = QAction("📡 RTSP Camera", self)
        rtsp_camera_action.setStatusTip("Kết nối RTSP Camera")
        rtsp_camera_action.triggered.connect(self.configure_rtsp_camera)
        file_menu.addAction(rtsp_camera_action)
        
        # Add File menu to toolbar
        file_btn = QPushButton("📁 File")
        file_btn.setMenu(file_menu)
        file_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 100, 100, 0.3);
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(120, 120, 120, 0.5);
            }
            QPushButton:pressed {
                background: rgba(140, 140, 140, 0.6);
            }
        """)
        toolbar.addWidget(file_btn)
        
        toolbar.addSeparator()

        # Video source info label
        self.source_label = QLabel("No source selected")
        self.source_label.setStyleSheet("padding: 5px; color: #b0b0b0; font-style: italic;")
        toolbar.addWidget(self.source_label)

        toolbar.addSeparator()

        # Start/Stop buttons
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_video)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: rgba(46, 204, 113, 0.4);
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background: rgba(46, 204, 113, 0.6);
            }
            QPushButton:pressed {
                background: rgba(46, 204, 113, 0.8);
            }
            QPushButton:disabled {
                background: rgba(80, 80, 80, 0.3);
                color: #999;
            }
        """)
        toolbar.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏸ Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: rgba(231, 76, 60, 0.4);
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background: rgba(231, 76, 60, 0.6);
            }
            QPushButton:pressed {
                background: rgba(231, 76, 60, 0.8);
            }
            QPushButton:disabled {
                background: rgba(80, 80, 80, 0.3);
                color: #999;
            }
        """)
        toolbar.addWidget(self.stop_btn)

        toolbar.addSeparator()

        # Logout button
        logout_action = QAction("🚪 Đăng xuất", self)
        logout_action.setStatusTip("Đăng xuất")
        logout_action.triggered.connect(self.logout)
        toolbar.addAction(logout_action)

    def create_video_panel(self):
        """Create video display panel"""
        panel = QGroupBox("Live View")
        panel.setStyleSheet("""
            QGroupBox {
                border: none;
                color: #e0e0e0;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Video label - không có border
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        
        # Set size policy để label mở rộng và fill đầy không gian có sẵn
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Không có border - màu nền xám
        self.video_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #404040, stop:1 #353535);
                border: none;
                border-radius: 8px;
            }
        """)
        self.video_label.setText("Video sẽ hiển thị ở đây\n\nĐang tải models...")

        font = QFont()
        font.setPointSize(14)
        self.video_label.setFont(font)
        # Update style sheet khi có text (chế độ chờ)
        self.video_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #404040, stop:1 #353535);
                color: #e0e0e0;
                border: none;
                border-radius: 8px;
            }
        """)

        layout.addWidget(self.video_label)

        # FPS label
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignRight)
        font_small = QFont()
        font_small.setPointSize(10)
        self.fps_label.setFont(font_small)
        self.fps_label.setStyleSheet("color: #b0b0b0; background: transparent; padding: 5px;")
        layout.addWidget(self.fps_label)

        panel.setLayout(layout)
        return panel

    def create_face_info_panel(self):
        """Create face info panel"""
        panel = QGroupBox("Kết quả nhận diện")
        panel.setStyleSheet("""
            QGroupBox {
                border: none;
                color: #e0e0e0;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Scroll area for faces
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(60, 60, 60, 0.5);
                width: 10px;
                border: none;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(150, 150, 150, 0.6);
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(180, 180, 180, 0.8);
            }
        """)

        self.faces_container = QWidget()
        self.faces_container.setStyleSheet("background: transparent;")
        self.faces_layout = QVBoxLayout()
        self.faces_layout.setContentsMargins(5, 5, 5, 5)
        self.faces_layout.setSpacing(8)
        self.faces_container.setLayout(self.faces_layout)

        scroll.setWidget(self.faces_container)
        layout.addWidget(scroll)

        panel.setLayout(layout)
        return panel

    def load_models_auto(self):
        """Auto-load models from database configuration"""
        try:
            self.status_bar.showMessage("Đang tải models...")
            log.info("Auto-loading models for Client Panel...")

            # Reload model config from database (in case Admin updated it)
            model_config = self.db.get_model_config()
            self.model_fd_path = model_config['model_fd_path']
            self.model_lm_path = model_config['model_lm_path']
            self.model_reid_path = model_config['model_reid_path']
            self.gallery_path = model_config['gallery_path']

            log.info(f"Loading models from config: FD={self.model_fd_path}, LM={self.model_lm_path}, ReID={self.model_reid_path}")

            # Check if model files exist
            if not all([
                Path(self.model_fd_path).exists(),
                Path(self.model_lm_path).exists(),
                Path(self.model_reid_path).exists()
            ]):
                error_msg = (
                    "Models không tìm thấy!\n\n"
                    f"Face Detection: {self.model_fd_path}\n"
                    f"Landmarks: {self.model_lm_path}\n"
                    f"Re-ID: {self.model_reid_path}\n\n"
                    "Vui lòng Admin cấu hình models trước."
                )
                self.status_bar.showMessage("⚠️ Models không tìm thấy! Admin cần cấu hình models.")
                self.video_label.setText(error_msg)
                return

            # Initialize OpenVINO Core
            self.core = Core()
            log.info(f"OpenVINO Version: {get_version()}")

            # Load Face Detector
            self.face_detector = FaceDetector(
                self.core,
                Path(self.model_fd_path),
                input_size=(0, 0),
                confidence_threshold=0.6,
                roi_scale_factor=1.15
            )
            self.face_detector.deploy("CPU")

            # Load Landmarks Detector
            self.landmarks_detector = LandmarksDetector(
                self.core,
                Path(self.model_lm_path)
            )
            self.landmarks_detector.deploy("CPU", 16)

            # Load Face Identifier
            self.face_identifier = FaceIdentifier(
                self.core,
                Path(self.model_reid_path),
                match_threshold=0.3,
                match_algo='HUNGARIAN'
            )
            self.face_identifier.deploy("CPU", 16)

            # Load Faces Database
            self.faces_database = FacesDatabase(
                self.gallery_path,
                self.face_identifier,
                self.landmarks_detector,
                self.face_detector,
                no_show=True
            )
            self.face_identifier.set_faces_database(self.faces_database)

            self.models_loaded = True
            # Update start button state (requires both models and source)
            self.update_start_button_state()

            self.status_bar.showMessage(
                f"✓ Models đã tải! Database: {len(self.faces_database)} identities"
            )
            self.video_label.setText("Models đã sẵn sàng!\n\nChọn nguồn video từ menu File, sau đó nhấn Start.")
            
            log.info(f"Client Panel: Models loaded successfully. Gallery: {len(self.faces_database)} identities")

        except Exception as e:
            log.error(f"Error loading models: {e}")
            self.status_bar.showMessage(f"✗ Lỗi khi tải models")
            self.video_label.setText(f"Lỗi khi tải models:\n{str(e)}")

    def start_video(self):
        """Start video processing"""
        if not self.models_loaded:
            self.status_bar.showMessage("⚠️ Vui lòng đợi models tải xong!")
            return
        
        if not self.video_source_configured or self.video_source is None:
            self.status_bar.showMessage("⚠️ Vui lòng chọn nguồn video từ menu File!")
            QMessageBox.warning(self, "Warning", "Please select a video source from File menu first!")
            return

        # Clear displayed crops khi start lại
        self.displayed_crops.clear()

        # Create and start video thread
        self.video_thread = VideoThread()
        self.video_thread.set_models(
            self.face_detector,
            self.landmarks_detector,
            self.face_identifier,
            self.faces_database
        )
        self.video_thread.set_database(self.db)  # Set database để load capture_interval
        self.video_thread.set_managers(
            self.events_manager,
            self.crops_manager,
            self.camera_id
        )  # Set managers để lưu events và crops
        self.video_thread.set_source(self.video_source)

        # Connect signals
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.face_detected_signal.connect(self.update_faces)
        self.video_thread.fps_signal.connect(self.update_fps)
        self.video_thread.error_signal.connect(self.handle_error)

        self.video_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_bar.showMessage("▶ Đang chạy...")

    def stop_video(self):
        """Stop video processing"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("⏸ Đã dừng")

        # Clear video display
        self.video_label.setText("Video đã dừng\n\nNhấn Start để tiếp tục")

    def update_image(self, frame):
        """Update video display - video sẽ fill đầy khung màu đỏ"""
        try:
            # Convert frame to QPixmap
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Scale to fill entire label (stretch to fit - IgnoreAspectRatio)
            # Điều này đảm bảo video fill đầy toàn bộ khung màu đỏ
            pixmap = QPixmap.fromImage(qt_image)
            label_size = self.video_label.size()
            
            # Kiểm tra kích thước hợp lệ để tránh crash khi resize
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = pixmap.scaled(
                    label_size,
                    Qt.IgnoreAspectRatio,  # Stretch để fill đầy khung
                    Qt.SmoothTransformation  # Smooth scaling
                )
                
                # Update style khi có video - không có border
                self.video_label.setStyleSheet("""
                    QLabel {
                        background: transparent;
                        border: none;
                        border-radius: 8px;
                    }
                """)
                self.video_label.setPixmap(scaled_pixmap)
            else:
                # Kích thước chưa hợp lệ, đợi resize event
                log.debug("Label size not valid yet, skipping update")
        except Exception as e:
            log.error(f"Error updating video image: {e}")
            # Fallback: giữ nguyên frame trước đó

    def get_gallery_image_path(self, label: str):
        """Get first image path from gallery for a label
        
        Tìm ảnh đầu tiên trong folder riêng của khách hàng: gallery/{base_name}/
        """
        if not self.gallery_path:
            return None
        
        gallery_dir = Path(self.gallery_path)
        if not gallery_dir.exists():
            return None
        
        # Lấy base_name từ label (ví dụ: "john-0" -> "john")
        base_name = label.split('-')[0] if '-' in label else label
        
        # Kiểm tra folder con của khách hàng
        customer_folder = gallery_dir / base_name
        if customer_folder.exists() and customer_folder.is_dir():
            # Tìm ảnh đầu tiên trong folder
            for ext in ['jpg', 'jpeg', 'png']:
                # Tìm image-*.jpg trước (format mới)
                images = sorted(customer_folder.glob(f"image-*.{ext}"))
                if images:
                    return str(images[0])
                # Fallback: tìm *.jpg
                images = sorted(customer_folder.glob(f"*.{ext}"))
                if images:
                    return str(images[0])
        
        # Fallback: tìm ảnh trực tiếp trong gallery (format cũ)
        for ext in ['jpg', 'jpeg', 'png']:
            pattern = f"{base_name}-*.{ext}"
            images = sorted(gallery_dir.glob(pattern))
            if images:
                return str(images[0])
        
        return None

    def update_faces(self, faces_info):
        """Update faces display - hiển thị real-time với interval 3 giây"""
        try:
            for face_info in faces_info:
                # Convert crop to QPixmap
                crop_img = face_info['crop']
                if crop_img is not None and crop_img.size > 0:
                    # Convert BGR to RGB
                    rgb_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_crop.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_crop.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)

                    # Xác định trạng thái: Khách quen hoặc Khách mới
                    is_known = face_info['label'] != "Unknown"
                    
                    # Lấy ảnh từ database nếu là khách quen
                    gallery_pixmap = None
                    if is_known:
                        gallery_image_path = self.get_gallery_image_path(face_info['label'])
                        if gallery_image_path and Path(gallery_image_path).exists():
                            try:
                                gallery_img = cv2.imread(gallery_image_path)
                                if gallery_img is not None and gallery_img.size > 0:
                                    rgb_gallery = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
                                    h_g, w_g, ch_g = rgb_gallery.shape
                                    bytes_per_line_g = ch_g * w_g
                                    qt_image_g = QImage(rgb_gallery.data, w_g, h_g, bytes_per_line_g, QImage.Format_RGB888)
                                    gallery_pixmap = QPixmap.fromImage(qt_image_g)
                            except Exception as e:
                                log.warning(f"Error loading gallery image: {e}")
                                gallery_pixmap = None

                    # Thêm vào displayed_crops (thêm vào đầu list)
                    self.displayed_crops.insert(0, {
                        'face_id': face_info['face_id'],
                        'pixmap': pixmap,
                        'gallery_pixmap': gallery_pixmap,  # Ảnh từ database
                        'timestamp': face_info['timestamp'],
                        'is_known': is_known,
                        'label': face_info['label'],
                        'confidence': face_info['confidence']
                    })

                    # Giới hạn số lượng crops hiển thị
                    if len(self.displayed_crops) > self.max_displayed_crops:
                        self.displayed_crops = self.displayed_crops[:self.max_displayed_crops]

            # Refresh display
            self.refresh_crops_display()

        except Exception as e:
            log.error(f"Error updating faces: {e}")

    def refresh_crops_display(self):
        """Refresh crops display từ displayed_crops list"""
        try:
            # Clear previous crops
            for i in reversed(range(self.faces_layout.count())):
                widget = self.faces_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            # Add crops từ displayed_crops
            for crop_data in self.displayed_crops:
                crop_widget = self.create_crop_widget_realtime(crop_data)
                self.faces_layout.addWidget(crop_widget)

            self.faces_layout.addStretch()

        except Exception as e:
            log.error(f"Error refreshing crops display: {e}")

    def create_crop_widget_realtime(self, crop_data):
        """Create widget for a crop với trạng thái Khách quen/Khách mới (real-time)"""
        widget = QWidget()
        # Horizontal layout: tất cả trong 1 hàng ngang
        face_layout = QHBoxLayout()
        face_layout.setSpacing(10)

        # Nếu là khách quen và có ảnh database, hiển thị cả 2 ảnh trong 1 hàng
        if crop_data['is_known'] and crop_data.get('gallery_pixmap'):
            # Ảnh từ database (bên trái)
            gallery_container = QVBoxLayout()
            gallery_label = QLabel()
            gallery_label.setPixmap(crop_data['gallery_pixmap'].scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            gallery_label.setAlignment(Qt.AlignCenter)
            gallery_label.setStyleSheet("border: none; border-radius: 5px;")
            gallery_container.addWidget(gallery_label)
            
            # Label cho ảnh database
            gallery_info_label = QLabel("📷 Database")
            gallery_info_label.setAlignment(Qt.AlignCenter)
            gallery_info_label.setStyleSheet("color: #d0d0d0; font-size: 9px; padding: 2px; background: transparent;")
            gallery_container.addWidget(gallery_info_label)
            
            gallery_widget = QWidget()
            gallery_widget.setStyleSheet("background: transparent;")
            gallery_widget.setLayout(gallery_container)
            face_layout.addWidget(gallery_widget)
            
            # Ảnh crop hiện tại (giữa)
            crop_container = QVBoxLayout()
            face_label = QLabel()
            face_label.setPixmap(crop_data['pixmap'].scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            face_label.setAlignment(Qt.AlignCenter)
            face_label.setStyleSheet("border: none; border-radius: 5px;")
            crop_container.addWidget(face_label)
            
            # Label cho ảnh crop hiện tại
            crop_info_label = QLabel("📸 Hiện tại")
            crop_info_label.setAlignment(Qt.AlignCenter)
            crop_info_label.setStyleSheet("color: #d0d0d0; font-size: 9px; padding: 2px; background: transparent;")
            crop_container.addWidget(crop_info_label)
            
            crop_widget = QWidget()
            crop_widget.setLayout(crop_container)
            face_layout.addWidget(crop_widget)
        else:
            # Chỉ hiển thị ảnh crop (khách mới hoặc không có ảnh database)
            face_label = QLabel()
            face_label.setPixmap(crop_data['pixmap'].scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            face_label.setAlignment(Qt.AlignCenter)
            face_layout.addWidget(face_label)

        # Xác định trạng thái và màu - nổi bật trên nền xám
        if crop_data['is_known']:
            status_text = f"Khách quen\n{crop_data['label']}"
            status_color = "#04ff00"  # Xanh lá sáng
            # Màu card cho khách quen: xanh lá nổi bật
            bg_color = "rgba(46, 204, 113, 0.25)"  # Tăng opacity để nổi bật hơn
        else:
            status_text = "Khách mới"
            status_color = "#ff9800"  # Cam sáng
            # Màu card cho khách mới: cam nổi bật
            bg_color = "rgba(255, 152, 0, 0.25)"  # Tăng opacity để nổi bật hơn
        
        # Info layout (vertical)
        info_layout = QVBoxLayout()
        
        # Status label
        text_label = QLabel(status_text)
        text_font = QFont()
        text_font.setPointSize(12)
        text_font.setBold(True)
        text_label.setFont(text_font)
        text_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        text_label.setStyleSheet(f"color: {status_color}; padding: 5px;")
        info_layout.addWidget(text_label)
        
        # Confidence score (nếu có)
        if crop_data.get('confidence', 0) > 0:
            conf_label = QLabel(f"Độ tin cậy: {crop_data['confidence']:.1f}%")
            conf_font = QFont()
            conf_font.setPointSize(9)
            conf_label.setFont(conf_font)
            conf_label.setStyleSheet("color: #ffffff; padding: 2px; background: transparent;")
            info_layout.addWidget(conf_label)
        
        # Timestamp label
        time_label = QLabel(f"⏰ {crop_data['timestamp']}")
        time_font = QFont()
        time_font.setPointSize(9)
        time_label.setFont(time_font)
        time_label.setStyleSheet("color: #e0e0e0; padding: 2px; background: transparent;")
        info_layout.addWidget(time_label)
        
        face_layout.addLayout(info_layout)
        face_layout.addStretch()

        widget.setLayout(face_layout)
        # Áp dụng màu động dựa trên trạng thái - không có border
        widget.setStyleSheet(f"""
            QWidget {{ 
                border: none; 
                border-radius: 8px; 
                padding: 10px; 
                background: {bg_color}; 
                margin-bottom: 5px;
            }}
        """)
        return widget

    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def handle_error(self, error_msg):
        """Handle error"""
        log.error(error_msg)
        self.status_bar.showMessage(f"✗ Lỗi: {error_msg}")
        self.stop_video()

    def check_model_config_update(self):
        """Check if Admin updated model configuration"""
        try:
            current_config = self.db.get_model_config()
            
            # Convert to string for comparison
            current_config_str = str(current_config)
            
            # If config changed and models are not loaded, try to reload
            if self.last_model_config != current_config_str:
                if not self.models_loaded:
                    log.info("Model configuration changed, reloading models...")
                    self.load_models_auto()
                elif self.last_model_config is not None:
                    # Config changed while models are loaded - need to reload
                    log.info("Model configuration updated by Admin, reloading models...")
                    was_running = self.stop_btn.isEnabled()
                    if was_running:
                        self.stop_video()
                    self.load_models_auto()
                    if was_running and self.models_loaded:
                        self.start_video()
                
                self.last_model_config = current_config_str
                
        except Exception as e:
            log.error(f"Error checking model config: {e}")

    def configure_webcam(self):
        """Configure webcam source"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Webcam")
        dialog.setMinimumSize(400, 200)
        
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        # Camera index input
        camera_index_input = QLineEdit("0")
        camera_index_input.setPlaceholderText("Camera index (usually 0)")
        form.addRow("Camera Index:", camera_index_input)
        
        # Name input
        name_input = QLineEdit("Webcam")
        name_input.setPlaceholderText("Source name")
        form.addRow("Name:", name_input)
        
        layout.addLayout(form)
        
        # Info label
        info_label = QLabel(
            "💡 Note: Camera index is usually 0 for default webcam.\n"
            "If you have multiple cameras, try 1, 2, etc."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(info_label)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                camera_index = int(camera_index_input.text().strip() or "0")
                source_name = name_input.text().strip() or "Webcam"
                
                self.video_source = camera_index
                self.video_source_type = 'webcam'
                self.video_source_name = source_name
                self.video_source_configured = True
                
                self.source_label.setText(f"📷 {source_name} (Index: {camera_index})")
                self.update_start_button_state()
                
                self.status_bar.showMessage(f"✓ Webcam configured: {source_name} (Index: {camera_index})")
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid camera index! Please enter a number.")
    
    def configure_local_video(self):
        """Configure local video file source"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Local Video")
        dialog.setMinimumSize(500, 250)
        
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        # Name input
        name_input = QLineEdit("")
        name_input.setPlaceholderText("Enter source name")
        form.addRow("Name:", name_input)
        
        # File path input
        file_path_input = QLineEdit()
        file_path_input.setPlaceholderText("No file selected")
        file_path_input.setReadOnly(True)
        
        browse_btn = QPushButton("📂 Browse...")
        browse_btn.clicked.connect(lambda: self.browse_video_file(file_path_input))
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(file_path_input)
        file_layout.addWidget(browse_btn)
        
        form.addRow("Video File:", file_layout)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            file_path = file_path_input.text().strip()
            source_name = name_input.text().strip()
            
            if not file_path:
                QMessageBox.warning(self, "Error", "Please select a video file!")
                return
            
            if not Path(file_path).exists():
                QMessageBox.warning(self, "Error", f"File not found: {file_path}")
                return
            
            if not source_name:
                source_name = Path(file_path).stem
            
            self.video_source = file_path
            self.video_source_type = 'local'
            self.video_source_name = source_name
            self.video_source_configured = True
            
            file_name = Path(file_path).name
            self.source_label.setText(f"🎬 {source_name} ({file_name})")
            self.update_start_button_state()
            
            self.status_bar.showMessage(f"✓ Local video configured: {source_name}")
    
    def browse_video_file(self, line_edit):
        """Browse for video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            ".",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*.*)"
        )
        if file_path:
            line_edit.setText(file_path)
    
    def configure_ip_camera(self):
        """Configure IP Camera source"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure IP Camera")
        dialog.setMinimumSize(500, 250)
        
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        # Name input
        name_input = QLineEdit("IP Camera")
        name_input.setPlaceholderText("Enter source name")
        form.addRow("Name:", name_input)
        
        # IP/URL input
        url_input = QLineEdit()
        url_input.setPlaceholderText("http://username:password@192.168.1.100:8080/video")
        form.addRow("Camera URL:", url_input)
        
        layout.addLayout(form)
        
        # Info label
        info_label = QLabel(
            "💡 Format examples:\n"
            "• http://192.168.1.100:8080/video\n"
            "• http://admin:password@192.168.1.100:8080/video\n"
            "• http://192.168.1.100/cgi-bin/video.cgi"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(info_label)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            url = url_input.text().strip()
            source_name = name_input.text().strip()
            
            if not url:
                QMessageBox.warning(self, "Error", "Please enter camera URL!")
                return
            
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            if not source_name:
                source_name = "IP Camera"
            
            self.video_source = url
            self.video_source_type = 'ip'
            self.video_source_name = source_name
            self.video_source_configured = True
            
            self.source_label.setText(f"🌐 {source_name}")
            self.update_start_button_state()
            
            self.status_bar.showMessage(f"✓ IP Camera configured: {source_name}")
    
    def configure_rtsp_camera(self):
        """Configure RTSP Camera source"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure RTSP Camera")
        dialog.setMinimumSize(500, 300)
        
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        # Name input
        name_input = QLineEdit("RTSP Camera")
        name_input.setPlaceholderText("Enter source name")
        form.addRow("Name:", name_input)
        
        # RTSP URL input
        rtsp_input = QLineEdit()
        rtsp_input.setPlaceholderText("rtsp://username:password@192.168.1.100:554/stream")
        form.addRow("RTSP URL:", rtsp_input)
        
        layout.addLayout(form)
        
        # Info label
        info_label = QLabel(
            "💡 RTSP URL format examples:\n"
            "• rtsp://192.168.1.100:554/stream\n"
            "• rtsp://admin:password@192.168.1.100:554/h264\n"
            "• rtsp://username:password@192.168.1.100:554/channel1/stream1"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(info_label)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            rtsp_url = rtsp_input.text().strip()
            source_name = name_input.text().strip()
            
            if not rtsp_url:
                QMessageBox.warning(self, "Error", "Please enter RTSP URL!")
                return
            
            if not rtsp_url.startswith('rtsp://'):
                rtsp_url = 'rtsp://' + rtsp_url
            
            if not source_name:
                source_name = "RTSP Camera"
            
            self.video_source = rtsp_url
            self.video_source_type = 'rtsp'
            self.video_source_name = source_name
            self.video_source_configured = True
            
            self.source_label.setText(f"📡 {source_name}")
            self.update_start_button_state()
            
            self.status_bar.showMessage(f"✓ RTSP Camera configured: {source_name}")
    
    def update_start_button_state(self):
        """Update Start button enabled state based on models and source configuration"""
        # Enable Start button only if models are loaded AND source is configured
        if self.models_loaded and self.video_source_configured:
            self.start_btn.setEnabled(True)
        else:
            self.start_btn.setEnabled(False)
    
    def logout(self):
        """Logout and return to login"""
        self.stop_video()
        self.model_config_timer.stop()
        self.logout_signal.emit()
        self.close()

    def showEvent(self, event):
        """Show fullscreen"""
        super().showEvent(event)
        try:
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen().availableGeometry()
            self.setGeometry(screen)
            self.showFullScreen()
        except:
            self.showFullScreen()
    
    def keyPressEvent(self, event):
        """Handle ESC key to exit fullscreen"""
        if event.key() == Qt.Key_Escape:
            # Toggle fullscreen on ESC
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.video_thread:
            self.video_thread.stop()
        self.model_config_timer.stop()
        event.accept()