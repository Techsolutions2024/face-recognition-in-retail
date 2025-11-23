#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webcam Registration Dialog - Đăng ký khuôn mặt bằng webcam
Với các động tác: Turn left, Turn right, Nod head, etc.
Mỗi động tác 3 giây, chụp nhiều frame
"""

import sys
import logging as log
from pathlib import Path
from typing import List, Dict, Optional
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

try:
    from utils import crop, FaceDetector, LandmarksDetector, FaceIdentifier, FacesDatabase
except ImportError:
    log.error("Cannot import face recognition modules")
    FaceDetector = None
    LandmarksDetector = None
    FaceIdentifier = None
    FacesDatabase = None


class WebcamRegistrationDialog(QDialog):
    """Dialog đăng ký khuôn mặt bằng webcam với các động tác"""
    
    # Các động tác (tiếng Anh)
    POSE_STEPS = [
        {"instruction": "Look straight", "duration": 3},
        {"instruction": "Turn left", "duration": 3},
        {"instruction": "Turn right", "duration": 3},
        {"instruction": "Nod head up", "duration": 3},
        {"instruction": "Nod head down", "duration": 3},
    ]
    
    # FPS để chụp frame (30 FPS = mỗi 33ms một frame)
    CAPTURE_FPS = 10  # Chụp 10 frame/giây = 30 frame trong 3 giây
    
    def __init__(self, customer_name: str, face_detector, landmarks_detector,
                 face_identifier, faces_database, gallery_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Webcam Registration - {customer_name}")
        self.setMinimumSize(900, 700)
        
        self.customer_name = customer_name
        self.face_detector = face_detector
        self.landmarks_detector = landmarks_detector
        self.face_identifier = face_identifier
        self.faces_database = faces_database
        self.gallery_path = gallery_path
        
        # Captured faces data
        self.captured_faces: List[Dict] = []
        
        # Camera
        self.cap = None
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        
        # Registration state
        self.registration_started = False
        self.current_step = 0
        self.step_start_time = 0
        self.step_timer = QTimer()
        self.step_timer.timeout.connect(self.update_step)
        
        # Capture state
        self.last_capture_time = 0
        self.capture_interval = 1.0 / self.CAPTURE_FPS  # Thời gian giữa các lần chụp
        
        # Current frame
        self.current_frame = None
        
        self.init_ui()
        self.start_camera()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("QLabel { background-color: black; border: 2px solid #ddd; }")
        layout.addWidget(self.video_label)
        
        # Instruction label
        self.instruction_label = QLabel("Ready to start registration")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        font_instruction = QFont()
        font_instruction.setPointSize(18)
        font_instruction.setBold(True)
        self.instruction_label.setFont(font_instruction)
        self.instruction_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        layout.addWidget(self.instruction_label)
        
        # Countdown/Progress label
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        font_progress = QFont()
        font_progress.setPointSize(16)
        font_progress.setBold(True)
        self.progress_label.setFont(font_progress)
        self.progress_label.setStyleSheet("color: #e74c3c; padding: 5px;")
        layout.addWidget(self.progress_label)
        
        # Info label
        self.info_label = QLabel(
            f"<b>Registering face for: {self.customer_name}</b><br><br>"
            f"Click <b>Start Registration</b> to begin.<br>"
            f"You will be guided through {len(self.POSE_STEPS)} poses."
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        font_info = QFont()
        font_info.setPointSize(10)
        self.info_label.setFont(font_info)
        layout.addWidget(self.info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Registration")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 30px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.start_btn.clicked.connect(self.start_registration)
        button_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("✗ Cancel")
        self.cancel_btn.setMinimumHeight(50)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 30px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.cancel_btn.clicked.connect(self.cancel_registration)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def start_camera(self):
        """Start camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Cannot open camera!")
                self.reject()
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.frame_timer.start(30)  # 30ms = ~33 FPS
            log.info("Camera started successfully")
        except Exception as e:
            log.error(f"Error starting camera: {e}")
            QMessageBox.critical(self, "Error", f"Cannot start camera: {e}")
            self.reject()
    
    def update_frame(self):
        """Update video frame"""
        if self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.current_frame = frame.copy()
        
        # Detect face
        if self.face_detector:
            rois = self.face_detector.infer((frame,))
            
            # Draw face detection
            display_frame = frame.copy()
            if len(rois) > 0:
                roi = rois[0]
                xmin = max(int(roi.position[0]), 0)
                ymin = max(int(roi.position[1]), 0)
                xmax = min(int(roi.position[0] + roi.size[0]), frame.shape[1])
                ymax = min(int(roi.position[1] + roi.size[1]), frame.shape[0])
                
                # Draw bounding box (green if registration active, yellow otherwise)
                color = (0, 255, 0) if self.registration_started else (0, 255, 255)
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Draw landmarks if available
                if self.landmarks_detector and len(rois) > 0:
                    landmarks = self.landmarks_detector.infer((frame, [roi]))
                    if landmarks:
                        lm = landmarks[0]
                        for point in lm:
                            x = int(xmin + roi.size[0] * point[0])
                            y = int(ymin + roi.size[1] * point[1])
                            cv2.circle(display_frame, (x, y), 2, (0, 255, 255), -1)
            else:
                # No face detected
                cv2.putText(display_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            display_frame = frame
        
        # Convert to QPixmap and display
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    def start_registration(self):
        """Start registration process"""
        # Use explicit None check because FacesDatabase can be empty (0 identities) but still valid
        models_loaded = (
            self.face_detector is not None and
            self.landmarks_detector is not None and
            self.face_identifier is not None and
            self.faces_database is not None
        )
        
        if not models_loaded:
            log.warning(f"Models check failed in start_registration: "
                       f"face_detector={self.face_detector is not None}, "
                       f"landmarks_detector={self.landmarks_detector is not None}, "
                       f"face_identifier={self.face_identifier is not None}, "
                       f"faces_database={self.faces_database is not None}")
            QMessageBox.warning(
                self,
                "Warning",
                "Models not loaded!\n\n"
                "Please load models in Admin Panel before registering faces."
            )
            return
        
        self.registration_started = True
        self.current_step = 0
        self.captured_faces = []
        self.start_btn.setEnabled(False)
        
        # Start first step
        self.start_step()
    
    def start_step(self):
        """Start current pose step"""
        if self.current_step >= len(self.POSE_STEPS):
            # All steps completed
            self.finish_registration()
            return
        
        step = self.POSE_STEPS[self.current_step]
        self.instruction_label.setText(step["instruction"])
        self.step_start_time = time.time()
        self.last_capture_time = 0
        
        # Start step timer (update every 100ms for smoother countdown)
        self.step_timer.start(100)
        self.update_step()
    
    def update_step(self):
        """Update current step progress"""
        if not self.registration_started:
            return
        
        step = self.POSE_STEPS[self.current_step]
        elapsed = time.time() - self.step_start_time
        remaining = step["duration"] - elapsed
        
        if remaining <= 0:
            # Step completed
            self.step_timer.stop()
            self.current_step += 1
            
            # Update info
            captured_count = len(self.captured_faces)
            self.info_label.setText(
                f"<b>Step {self.current_step}/{len(self.POSE_STEPS)} completed!</b><br>"
                f"Captured {captured_count} faces so far."
            )
            
            # Start next step after short delay
            QTimer.singleShot(1000, self.start_step)
        else:
            # Update countdown
            self.progress_label.setText(f"{int(remaining) + 1} seconds remaining")
            
            # Capture frame periodically
            current_time = time.time()
            if current_time - self.last_capture_time >= self.capture_interval:
                self.capture_frame()
                self.last_capture_time = current_time
    
    def capture_frame(self):
        """Capture current frame and process face"""
        if self.current_frame is None:
            return
        
        if not self.face_detector:
            return
        
        try:
            # Detect face
            rois = self.face_detector.infer((self.current_frame,))
            if len(rois) == 0:
                return
            
            roi = rois[0]
            
            # Detect landmarks
            landmarks = self.landmarks_detector.infer((self.current_frame, [roi]))
            if len(landmarks) == 0:
                return
            
            lm = landmarks[0]
            
            # Crop face
            face_img = crop(self.current_frame, roi)
            
            # Extract descriptor
            self.face_identifier.start_async(self.current_frame, [roi], [lm])
            descriptor = self.face_identifier.get_descriptors()[0]
            
            # Store captured face data
            self.captured_faces.append({
                'image': face_img,
                'descriptor': descriptor,
                'pose': self.POSE_STEPS[self.current_step]["instruction"]
            })
            
            log.debug(f"Captured frame for pose: {self.POSE_STEPS[self.current_step]['instruction']}")
            
        except Exception as e:
            log.error(f"Error capturing frame: {e}")
    
    def finish_registration(self):
        """Finish registration and save faces"""
        if len(self.captured_faces) == 0:
            QMessageBox.warning(self, "Warning", "No faces captured!")
            self.reject()
            return
        
        try:
            # Update UI
            self.instruction_label.setText("Saving faces...")
            self.progress_label.setText(f"Saving {len(self.captured_faces)} faces...")
            self.info_label.setText("Please wait...")
            
            # Save all captured faces to gallery
            saved_count = 0
            for face_data in self.captured_faces:
                match_index = self.faces_database.dump_faces(
                    face_data['image'],
                    face_data['descriptor'],
                    self.customer_name
                )
                if match_index >= 0:
                    saved_count += 1
            
            # Get face_id from database
            identity = None
            if len(self.faces_database) > 0:
                # Find the identity matching customer name
                customer_base_name = self.customer_name.lower().replace(' ', '-')
                for i in range(len(self.faces_database)):
                    if self.faces_database[i].label.lower() == customer_base_name:
                        identity = self.faces_database[i]
                        break
            
            face_id = None
            if identity:
                face_id = f"{identity.label}-0"
            else:
                face_id = f"{customer_base_name}-0"
            
            QMessageBox.information(
                self,
                "Success!",
                f"Registration completed!\n\n"
                f"• Customer: {self.customer_name}\n"
                f"• Captured: {saved_count} faces\n"
                f"• Face ID: {face_id}\n\n"
                f"Faces saved to gallery successfully!"
            )
            
            self.accept()
            
        except Exception as e:
            log.error(f"Error finishing registration: {e}")
            QMessageBox.critical(self, "Error", f"Error saving faces: {e}")
            self.reject()
    
    def cancel_registration(self):
        """Cancel registration"""
        if self.registration_started and len(self.captured_faces) > 0:
            reply = QMessageBox.question(
                self,
                "Confirm",
                f"You have captured {len(self.captured_faces)} faces.\n"
                f"Are you sure you want to cancel?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self.stop_camera()
        self.reject()
    
    def stop_camera(self):
        """Stop camera"""
        if self.frame_timer.isActive():
            self.frame_timer.stop()
        if self.step_timer.isActive():
            self.step_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def closeEvent(self, event):
        """Handle close event"""
        self.stop_camera()
        event.accept()
    
    def get_face_id(self) -> Optional[str]:
        """Get face_id after registration"""
        if len(self.captured_faces) == 0:
            return None
        
        customer_base_name = self.customer_name.lower().replace(' ', '-')
        if len(self.faces_database) > 0:
            for i in range(len(self.faces_database)):
                if self.faces_database[i].label.lower() == customer_base_name:
                    return f"{self.faces_database[i].label}-0"
        
        return f"{customer_base_name}-0"
