#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Management Dialog - Quản lý khách hàng (Admin only)
Hỗ trợ thêm/sửa/xóa khách hàng, upload ảnh khuôn mặt thủ công
"""

import sys
import os
import logging as log
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QFormLayout, QLineEdit,
    QComboBox, QDialogButtonBox, QAbstractItemView, QLabel, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont

from database import Database
from models import CustomerSegment

try:
    from utils import crop, FaceDetector, LandmarksDetector, FaceIdentifier, FacesDatabase
except ImportError:
    log.error("Cannot import face recognition modules")
    FaceDetector = None
    LandmarksDetector = None
    FaceIdentifier = None
    FacesDatabase = None

try:
    from webcam_registration_dialog import WebcamRegistrationDialog
except ImportError:
    log.error("Cannot import webcam registration dialog")
    WebcamRegistrationDialog = None


class CustomerManagementDialog(QDialog):
    """Dialog quản lý khách hàng"""

    def __init__(self, db: Database, parent=None,
                 face_detector=None, landmarks_detector=None,
                 face_identifier=None, faces_database=None, gallery_path=None):
        super().__init__(parent)
        self.db = db
        self.face_detector = face_detector
        self.landmarks_detector = landmarks_detector
        self.face_identifier = face_identifier
        self.faces_database = faces_database
        self.gallery_path = gallery_path or "./gallery"
        
        self.setWindowTitle("Quản lý Khách hàng")
        self.setMinimumSize(900, 600)
        self.init_ui()
        self.load_customers()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["ID", "Face ID", "Name", "Segment", "Gallery Images", "Visits", "Last Visit"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # Set column widths
        self.table.setColumnWidth(0, 50)   # ID
        self.table.setColumnWidth(1, 120)  # Face ID
        self.table.setColumnWidth(2, 150)  # Name
        self.table.setColumnWidth(3, 100)  # Segment
        self.table.setColumnWidth(4, 120)  # Gallery Images
        self.table.setColumnWidth(5, 80)   # Visits
        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()

        self.add_btn = QPushButton("➕ Thêm")
        self.add_btn.clicked.connect(self.add_customer)
        button_layout.addWidget(self.add_btn)

        self.edit_btn = QPushButton("✏️ Sửa")
        self.edit_btn.clicked.connect(self.edit_customer)
        button_layout.addWidget(self.edit_btn)

        self.delete_btn = QPushButton("🗑️ Xóa")
        self.delete_btn.clicked.connect(self.delete_customer)
        button_layout.addWidget(self.delete_btn)

        button_layout.addStretch()

        self.refresh_btn = QPushButton("🔄 Làm mới")
        self.refresh_btn.clicked.connect(self.load_customers)
        button_layout.addWidget(self.refresh_btn)

        self.close_btn = QPushButton("Đóng")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_gallery_image_count(self, face_id: str) -> int:
        """Get number of gallery images for a face_id
        
        Tìm ảnh trong folder riêng của khách hàng: gallery/{base_name}/
        """
        if not self.gallery_path:
            return 0
        
        gallery_dir = Path(self.gallery_path)
        if not gallery_dir.exists():
            return 0
        
        # Lấy base_name từ face_id (ví dụ: "john-0" -> "john")
        base_name = face_id.split('-')[0] if '-' in face_id else face_id
        
        # Kiểm tra folder con của khách hàng
        customer_folder = gallery_dir / base_name
        if customer_folder.exists() and customer_folder.is_dir():
            # Đếm ảnh trong folder con
            count = 0
            for ext in ['jpg', 'jpeg', 'png']:
                count += len(list(customer_folder.glob(f"*.{ext}")))
                count += len(list(customer_folder.glob(f"image-*.{ext}")))
            return count
        
        # Fallback: tìm ảnh trực tiếp trong gallery (format cũ)
        count = 0
        for ext in ['jpg', 'jpeg', 'png']:
            pattern = f"{base_name}-*.{ext}"
            count += len(list(gallery_dir.glob(pattern)))
        
        return count

    def load_customers(self):
        """Load customers from database"""
        customers = self.db.get_all_customers()
        self.table.setRowCount(len(customers))

        for row, customer in enumerate(customers):
            self.table.setItem(row, 0, QTableWidgetItem(str(customer.id)))
            self.table.setItem(row, 1, QTableWidgetItem(customer.face_id))
            self.table.setItem(row, 2, QTableWidgetItem(customer.name))
            self.table.setItem(row, 3, QTableWidgetItem(customer.segment))
            
            # Gallery images count
            gallery_count = self.get_gallery_image_count(customer.face_id)
            self.table.setItem(row, 4, QTableWidgetItem(str(gallery_count)))
            
            self.table.setItem(row, 5, QTableWidgetItem(str(customer.total_visits)))
            last_visit = customer.last_visit_date.strftime("%Y-%m-%d %H:%M") if customer.last_visit_date else "Never"
            self.table.setItem(row, 6, QTableWidgetItem(last_visit))

    def process_uploaded_image(self, image_path: str, customer_name: str) -> Optional[str]:
        """
        Process uploaded image: detect face, extract descriptor, save to gallery
        
        Returns:
            face_id if successful, None otherwise
        """
        # Use explicit None check because FacesDatabase can be empty (0 identities) but still valid
        models_loaded = (
            self.face_detector is not None and
            self.landmarks_detector is not None and
            self.face_identifier is not None and
            self.faces_database is not None
        )
        
        if not models_loaded:
            QMessageBox.warning(
                self,
                "Cảnh báo",
                "Models chưa được load!\n\n"
                "Vui lòng load models trong Admin Panel trước khi thêm khách hàng với ảnh."
            )
            return None

        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                QMessageBox.warning(self, "Lỗi", "Không thể đọc file ảnh!")
                return None

            # Detect faces
            rois = self.face_detector.infer((image,))
            if len(rois) == 0:
                QMessageBox.warning(
                    self,
                    "Cảnh báo",
                    "Không tìm thấy khuôn mặt trong ảnh!\n\n"
                    "Vui lòng chọn ảnh có khuôn mặt rõ ràng."
                )
                return None

            if len(rois) > 1:
                QMessageBox.warning(
                    self,
                    "Cảnh báo",
                    f"Tìm thấy {len(rois)} khuôn mặt trong ảnh!\n\n"
                    "Chỉ sử dụng khuôn mặt đầu tiên."
                )

            # Get first face
            roi = rois[0]

            # Detect landmarks
            landmarks = self.landmarks_detector.infer((image, [roi]))
            if len(landmarks) == 0:
                QMessageBox.warning(self, "Lỗi", "Không thể detect landmarks!")
                return None

            lm = landmarks[0]

            # Crop face
            face_img = crop(image, roi)

            # Extract descriptor
            self.face_identifier.start_async(image, [roi], [lm])
            descriptor = self.face_identifier.get_descriptors()[0]

            # Save to gallery (this will create face_id automatically)
            match_index = self.faces_database.dump_faces(
                face_img,
                descriptor,
                customer_name
            )

            # Get face_id from database
            if match_index >= 0 and match_index < len(self.faces_database):
                identity = self.faces_database[match_index]
                base_name = identity.label
                
                # Ảnh đã được lưu vào folder con, dùng base_name làm face_id
                # Format face_id: base_name-0, base_name-1, etc. nhưng với folder con thì chỉ cần base_name-0
                gallery_dir = Path(self.gallery_path)
                customer_folder = gallery_dir / base_name
                
                if customer_folder.exists():
                    # Đếm số ảnh trong folder con để tạo face_id
                    image_count = self.get_gallery_image_count(f"{base_name}-0")
                    # Trả về face_id với index là số ảnh - 1 (vì vừa thêm 1 ảnh)
                    return f"{base_name}-{image_count - 1}"
                else:
                    # Fallback: tìm ảnh trực tiếp trong gallery
                    for ext in ['jpg', 'jpeg', 'png']:
                        pattern = f"{base_name}-*.{ext}"
                        files = sorted(gallery_dir.glob(pattern))
                        if files:
                            latest_file = files[-1]
                            face_id = latest_file.stem
                            return face_id

            # Fallback: generate face_id from name
            return f"{customer_name.lower().replace(' ', '-')}-0"

        except Exception as e:
            log.error(f"Error processing uploaded image: {e}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")
            return None

    def add_customer(self):
        """Add new customer with optional image upload"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Thêm khách hàng mới")
        dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout()

        # Form layout
        form_layout = QFormLayout()

        # Name input
        name_input = QLineEdit()
        name_input.setPlaceholderText("Nhập tên khách hàng...")
        form_layout.addRow("Tên khách hàng:", name_input)

        # Segment
        segment_combo = QComboBox()
        segment_combo.addItems([
            CustomerSegment.REGULAR,
            CustomerSegment.VIP,
            CustomerSegment.NEW,
            CustomerSegment.BLACKLIST
        ])
        form_layout.addRow("Phân loại:", segment_combo)

        # Image upload section
        image_label = QLabel("Ảnh khuôn mặt:")
        form_layout.addRow(image_label)

        image_layout = QVBoxLayout()
        self.uploaded_image_path = None
        self.face_id_from_webcam = None
        
        self.image_preview_label = QLabel("Chưa chọn ảnh")
        self.image_preview_label.setMinimumSize(200, 200)
        self.image_preview_label.setStyleSheet(
            "QLabel { border: 2px dashed #ccc; background-color: #f5f5f5; }"
        )
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_preview_label)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        
        upload_btn = QPushButton("📷 Choose Image...")
        upload_btn.setMinimumHeight(40)
        upload_btn.clicked.connect(lambda: self.select_image(dialog, name_input))
        button_layout.addWidget(upload_btn)
        
        webcam_btn = QPushButton("📹 Use Webcam")
        webcam_btn.setMinimumHeight(40)
        webcam_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        webcam_btn.clicked.connect(lambda: self.use_webcam(dialog, name_input))
        button_layout.addWidget(webcam_btn)
        
        image_layout.addLayout(button_layout)
        form_layout.addRow(image_layout)

        # Info label
        info_label = QLabel(
            "💡 Instructions:\n"
            "• Enter customer name and select segment\n"
            "• (Optional) Upload face image OR use webcam to register\n"
            "• If no image/webcam, Face ID will be auto-generated\n"
            "• Image must have a clear face, only 1 person\n"
            "• Webcam: Follow poses (turn left, turn right, nod head, etc.)"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 10px; background-color: #f5f5f5; border-radius: 5px;")
        form_layout.addRow(info_label)

        layout.addLayout(form_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            name = name_input.text().strip()
            segment = segment_combo.currentText()

            if not name:
                QMessageBox.warning(self, "Warning", "Please enter customer name!")
                return

            try:
                face_id = None
                
                # Use face_id from webcam if available
                if self.face_id_from_webcam:
                    face_id = self.face_id_from_webcam
                # Process image if uploaded
                elif self.uploaded_image_path:
                    face_id = self.process_uploaded_image(self.uploaded_image_path, name)
                    if not face_id:
                        # User cancelled or error occurred
                        return
                else:
                    # Generate face_id from name
                    face_id = f"{name.lower().replace(' ', '-')}-0"

                # Add customer to database
                customer_id = self.db.add_customer(face_id, name, segment)
                
                message = f"Customer added successfully!\n\n"
                message += f"• Name: {name}\n"
                message += f"• Face ID: {face_id}\n"
                message += f"• Segment: {segment}\n"
                if self.uploaded_image_path:
                    message += f"• Image registered to gallery"
                elif self.face_id_from_webcam:
                    message += f"• Faces registered via webcam"
                
                QMessageBox.information(self, "Success", message)
                self.load_customers()
                
            except Exception as e:
                log.error(f"Error adding customer: {e}")
                QMessageBox.critical(self, "Error", f"Error adding customer: {str(e)}")

    def select_image(self, parent_dialog, name_input):
        """Select image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            parent_dialog,
            "Choose Face Image",
            ".",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*.*)"
        )
        
        if file_path:
            self.uploaded_image_path = file_path
            self.face_id_from_webcam = None  # Clear webcam face_id
            # Show preview
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    200, 200,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_preview_label.setPixmap(scaled_pixmap)
                self.image_preview_label.setText("")
                # Reset style
                self.image_preview_label.setStyleSheet(
                    "QLabel { border: 2px dashed #ccc; background-color: #f5f5f5; }"
                )
            else:
                self.image_preview_label.setText("Cannot load image")
                self.uploaded_image_path = None
    
    def use_webcam(self, parent_dialog, name_input):
        """Open webcam registration dialog"""
        # Check if models are loaded
        models_status = {
            'face_detector': self.face_detector is not None,
            'landmarks_detector': self.landmarks_detector is not None,
            'face_identifier': self.face_identifier is not None,
            'faces_database': self.faces_database is not None
        }
        
        if not all(models_status.values()):
            log.warning(f"Models check failed in use_webcam: {models_status}")
            failed_models = [name for name, status in models_status.items() if not status]
            QMessageBox.warning(
                parent_dialog,
                "Warning",
                f"Models not loaded!\n\n"
                f"Missing models: {', '.join(failed_models)}\n\n"
                f"Please load models in Admin Panel before using webcam registration."
            )
            return
        
        # Get customer name from name_input
        customer_name = name_input.text().strip()
        if not customer_name:
            QMessageBox.warning(
                parent_dialog,
                "Warning",
                "Please enter customer name first!"
            )
            return
        
        # Check if WebcamRegistrationDialog is available
        if WebcamRegistrationDialog is None:
            QMessageBox.warning(
                parent_dialog,
                "Error",
                "Webcam registration dialog is not available!"
            )
            return
        
        # Open webcam registration dialog
        try:
            webcam_dialog = WebcamRegistrationDialog(
                customer_name,
                self.face_detector,
                self.landmarks_detector,
                self.face_identifier,
                self.faces_database,
                self.gallery_path,
                parent_dialog
            )
            
            if webcam_dialog.exec_() == QDialog.Accepted:
                # Get face_id from webcam registration
                face_id = webcam_dialog.get_face_id()
                if face_id:
                    self.face_id_from_webcam = face_id
                    self.uploaded_image_path = None  # Clear uploaded image
                    
                    # Update preview
                    self.image_preview_label.setText("✓ Registered via Webcam")
                    self.image_preview_label.setStyleSheet(
                        "QLabel { border: 2px solid #27ae60; background-color: #d5f4e6; color: #27ae60; font-weight: bold; padding: 10px; }"
                    )
        except Exception as e:
            log.error(f"Error in webcam registration: {e}")
            QMessageBox.critical(parent_dialog, "Error", f"Error in webcam registration: {str(e)}")

    def edit_customer(self):
        """Edit selected customer"""
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn khách hàng để sửa!")
            return

        row = selected[0].row()
        customer_id = int(self.table.item(row, 0).text())
        customer = self.db.get_customer(customer_id)

        if not customer:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy khách hàng!")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Sửa thông tin khách hàng")
        dialog.setMinimumSize(400, 300)

        layout = QVBoxLayout()

        # Form layout
        form_layout = QFormLayout()

        # Name
        name_input = QLineEdit(customer.name)
        form_layout.addRow("Tên:", name_input)

        # Segment
        segment_combo = QComboBox()
        segment_combo.addItems([
            CustomerSegment.REGULAR,
            CustomerSegment.VIP,
            CustomerSegment.NEW,
            CustomerSegment.BLACKLIST
        ])
        segment_combo.setCurrentText(customer.segment)
        form_layout.addRow("Phân loại:", segment_combo)

        # Face ID (read-only)
        face_id_label = QLabel(customer.face_id)
        face_id_label.setStyleSheet("color: #666;")
        form_layout.addRow("Face ID:", face_id_label)

        # Gallery images count
        gallery_count = self.get_gallery_image_count(customer.face_id)
        gallery_count_label = QLabel(str(gallery_count))
        gallery_count_label.setStyleSheet("color: #666;")
        form_layout.addRow("Số ảnh trong gallery:", gallery_count_label)

        layout.addLayout(form_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            name = name_input.text().strip()
            segment = segment_combo.currentText()

            if not name:
                QMessageBox.warning(self, "Cảnh báo", "Tên không được để trống!")
                return

            try:
                self.db.update_customer(customer_id, name=name, segment=segment)
                QMessageBox.information(self, "Thành công", "Đã cập nhật thông tin khách hàng!")
                self.load_customers()
            except Exception as e:
                log.error(f"Error updating customer: {e}")
                QMessageBox.critical(self, "Lỗi", f"Lỗi khi cập nhật: {str(e)}")

    def delete_gallery_files(self, face_id: str):
        """Delete gallery image files for a face_id
        
        Xóa folder con của khách hàng: gallery/{base_name}/
        """
        if not self.gallery_path:
            return 0
        
        gallery_dir = Path(self.gallery_path)
        if not gallery_dir.exists():
            return 0
        
        deleted_count = 0
        base_name = face_id.split('-')[0] if '-' in face_id else face_id
        
        # Xóa folder con của khách hàng (nếu có)
        customer_folder = gallery_dir / base_name
        if customer_folder.exists() and customer_folder.is_dir():
            try:
                # Đếm số file trước khi xóa
                for ext in ['jpg', 'jpeg', 'png']:
                    deleted_count += len(list(customer_folder.glob(f"*.{ext}")))
                
                # Xóa toàn bộ folder
                import shutil
                shutil.rmtree(customer_folder)
                log.info(f"Deleted gallery folder: {customer_folder}")
            except Exception as e:
                log.error(f"Error deleting gallery folder {customer_folder}: {e}")
        
        # Fallback: xóa ảnh trực tiếp trong gallery (format cũ)
        for ext in ['jpg', 'jpeg', 'png']:
            pattern = f"{base_name}-*.{ext}"
            for file_path in gallery_dir.glob(pattern):
                try:
                    file_path.unlink()
                    deleted_count += 1
                    log.info(f"Deleted gallery file: {file_path}")
                except Exception as e:
                    log.error(f"Error deleting gallery file {file_path}: {e}")
        
        return deleted_count

    def delete_customer(self):
        """Delete selected customer and all related data"""
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn khách hàng để xóa!")
            return

        row = selected[0].row()
        customer_id = int(self.table.item(row, 0).text())
        customer_name = self.table.item(row, 2).text()
        face_id = self.table.item(row, 1).text()

        # Get counts for confirmation message
        gallery_count = self.get_gallery_image_count(face_id)
        
        confirm_msg = (
            f"Bạn có chắc muốn xóa khách hàng '{customer_name}'?\n\n"
            f"⚠️ CẢNH BÁO: Hành động này sẽ xóa:\n"
            f"• Thông tin khách hàng\n"
            f"• Tất cả events liên quan\n"
            f"• Tất cả crops (ảnh đã chụp)\n"
            f"• Tất cả visits\n"
        )
        
        if gallery_count > 0:
            confirm_msg += f"• {gallery_count} ảnh trong gallery\n"
        
        confirm_msg += "\nHành động này KHÔNG THỂ hoàn tác!"

        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Delete gallery files first
                deleted_files = 0
                if gallery_count > 0:
                    deleted_files = self.delete_gallery_files(face_id)
                
                # Delete from database (cascade delete handled in database.py)
                self.db.delete_customer(customer_id)
                
                success_msg = f"Đã xóa khách hàng '{customer_name}'!\n\n"
                success_msg += f"• Đã xóa khỏi database\n"
                if deleted_files > 0:
                    success_msg += f"• Đã xóa {deleted_files} ảnh trong gallery\n"
                
                QMessageBox.information(self, "Thành công", success_msg)
                self.load_customers()
                
            except Exception as e:
                log.error(f"Error deleting customer: {e}")
                QMessageBox.critical(self, "Lỗi", f"Lỗi khi xóa khách hàng: {str(e)}")
