#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Admin Panel - Giao diện quản trị hiện đại với Sidebar Navigation
Full screen, với sidebar menu chuyên nghiệp
"""

import sys
import logging as log
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QDialog, QDialogButtonBox, QFormLayout,
    QLineEdit, QDoubleSpinBox, QMessageBox, QFrame,
    QGridLayout, QScrollArea, QSpinBox, QComboBox, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QPalette, QColor

from database import Database
from models import CustomerSegment, EventType

# Import các dialog cần thiết
from user_management_dialog import UserManagementDialog
from customer_management_dialog import CustomerManagementDialog

# Import OpenVINO and face recognition modules
try:
    from openvino import Core, get_version
except ImportError:
    print("Vui lòng cài đặt OpenVINO: pip install openvino")
    Core = None
    get_version = None

try:
    from utils import (
        FaceDetector,
        LandmarksDetector,
        FaceIdentifier,
        FacesDatabase
    )
except ImportError:
    print("Không thể import utils. Vui lòng đảm bảo file utils.py tồn tại.")
    FaceDetector = None
    LandmarksDetector = None
    FaceIdentifier = None
    FacesDatabase = None


class ModernCard(QFrame):
    """Modern card widget với shadow effect"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        if title:
            title_label = QLabel(title)
            title_font = QFont()
            title_font.setPointSize(14)
            title_font.setBold(True)
            title_label.setFont(title_font)
            title_label.setStyleSheet("color: #1e293b; border: none;")
            layout.addWidget(title_label)
        
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(12)
        layout.addLayout(self.content_layout)
        
        self.setLayout(layout)
    
    def add_widget(self, widget):
        """Add widget to card content"""
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        """Add layout to card content"""
        self.content_layout.addLayout(layout)


class StatsCard(QFrame):
    """Statistics card widget"""
    
    def __init__(self, icon="", title="", value="0", color="#6366f1", parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {color}, stop:1 {self.adjust_color(color, 0.8)});
                border-radius: 12px;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)
        
        # Icon + Title row
        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("color: white; font-size: 32px; border: none;")
        top_row.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: rgba(255,255,255,0.9); font-size: 13px; font-weight: 600; border: none;")
        top_row.addWidget(title_label)
        top_row.addStretch()
        
        layout.addLayout(top_row)
        layout.addStretch()
        
        # Value
        self.value_label = QLabel(value)
        value_font = QFont()
        value_font.setPointSize(28)
        value_font.setBold(True)
        self.value_label.setFont(value_font)
        self.value_label.setStyleSheet("color: white; border: none;")
        layout.addWidget(self.value_label)
        
        self.setLayout(layout)
    
    def adjust_color(self, color, factor):
        """Adjust color brightness"""
        return color
    
    def set_value(self, value):
        """Update value"""
        self.value_label.setText(str(value))


class SidebarButton(QPushButton):
    """Custom sidebar navigation button"""
    
    def __init__(self, icon, text, parent=None):
        super().__init__(parent)
        self.icon_text = icon
        self.label_text = text
        self.is_active = False
        
        self.setText(f"{icon}  {text}")
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        self.update_style()
    
    def set_active(self, active):
        """Set button active state"""
        self.is_active = active
        self.update_style()
    
    def update_style(self):
        """Update button style based on state"""
        if self.is_active:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #6366f1;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 16px;
                    text-align: left;
                    font-size: 14px;
                    font-weight: 600;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #64748b;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 16px;
                    text-align: left;
                    font-size: 14px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #f1f5f9;
                    color: #1e293b;
                }
            """)


class EventDetailDialog(QDialog):
    """Dialog hiển thị chi tiết event với ảnh"""
    
    def __init__(self, event, db, parent=None):
        super().__init__(parent)
        self.event = event
        self.db = db
        
        self.setWindowTitle(f"Chi tiết Event #{event.id}")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: #f8fafc;")
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Header
        header = QLabel(f"Event #{event.id}")
        header_font = QFont()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet("color: #0f172a;")
        layout.addWidget(header)
        
        # Image card
        image_card = ModernCard()
        
        # Load crop image
        try:
            all_crops = self.db.get_recent_crops(limit=1000)
            # Tìm crop theo event_id trước
            event_crops = [c for c in all_crops if hasattr(c, 'event_id') and c.event_id == event.id]
            
            # Nếu không tìm thấy theo event_id, tìm theo customer_id và timestamp (fallback)
            if not event_crops and event.customer_id:
                event_time = event.timestamp
                if event_time:
                    # Tìm crop của cùng customer trong khoảng thời gian gần event (trong vòng 10 phút)
                    from datetime import timedelta
                    time_window_start = event_time - timedelta(minutes=10)
                    time_window_end = event_time + timedelta(minutes=10)
                    event_crops = [
                        c for c in all_crops 
                        if hasattr(c, 'customer_id') and c.customer_id == event.customer_id
                        and hasattr(c, 'timestamp') and c.timestamp
                        and time_window_start <= c.timestamp <= time_window_end
                    ]
            
            if event_crops and len(event_crops) > 0:
                crop = event_crops[0]
                file_path = Path(crop.file_path)
                if file_path.exists():
                    pixmap = QPixmap(str(file_path))
                    if not pixmap.isNull():
                        image_label = QLabel()
                        scaled = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        image_label.setPixmap(scaled)
                        image_label.setAlignment(Qt.AlignCenter)
                        image_card.add_widget(image_label)
                    else:
                        no_image = QLabel("Không có ảnh")
                        no_image.setAlignment(Qt.AlignCenter)
                        image_card.add_widget(no_image)
                else:
                    no_image = QLabel("Không tìm thấy file ảnh")
                    no_image.setAlignment(Qt.AlignCenter)
                    image_card.add_widget(no_image)
            else:
                no_crop = QLabel("Không có crop cho event này")
                no_crop.setAlignment(Qt.AlignCenter)
                image_card.add_widget(no_crop)
        except Exception as e:
            log.error(f"Error loading crops for event {event.id}: {e}")
            error_label = QLabel(f"Lỗi tải ảnh: {str(e)}")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: #ef4444;")
            image_card.add_widget(error_label)
        
        layout.addWidget(image_card)
        
        # Info card
        info_card = ModernCard()
        info_form = QFormLayout()
        info_form.setSpacing(12)
        
        # Event details
        # Format event type
        event_type_display = event.event_type
        if event.event_type == EventType.VIP_DETECTED:
            event_type_display = "VIP"
        elif event.event_type == EventType.NEW_CUSTOMER:
            event_type_display = "Khách mới"
        elif event.event_type == EventType.REGULAR_VISIT:
            event_type_display = "Khách thường"
        elif event.event_type == EventType.BLACKLIST:
            event_type_display = "Blacklist"
        
        # Get confidence summary from metadata
        confidence_display = f"{event.confidence:.1f}%"
        duration_display = "N/A"
        if event.metadata and isinstance(event.metadata, dict):
            if 'confidence_avg' in event.metadata:
                confidence_display = f"{event.metadata['confidence_avg']:.1f}% (avg)"
            elif 'confidences' in event.metadata and len(event.metadata['confidences']) > 0:
                avg_conf = sum(event.metadata['confidences']) / len(event.metadata['confidences'])
                confidence_display = f"{avg_conf:.1f}% (avg)"
            
            if 'duration_formatted' in event.metadata:
                duration_display = event.metadata['duration_formatted']
            elif 'duration_seconds' in event.metadata:
                duration_seconds = event.metadata['duration_seconds']
                duration_display = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
        
        details = [
            ("Khách hàng:", event.customer_name),
            ("Loại:", event_type_display),
            ("Camera:", f"#{event.camera_id}" if event.camera_id else "N/A"),
            ("Độ tin cậy:", confidence_display),
            ("Thời lượng:", duration_display),
            ("Thời gian:", event.timestamp.strftime("%Y-%m-%d %H:%M:%S") if event.timestamp else "N/A"),
        ]
        
        for label_text, value_text in details:
            label = QLabel(label_text)
            label.setStyleSheet("color: #64748b; font-weight: 600; font-size: 13px;")
            
            value = QLabel(str(value_text))
            value.setStyleSheet("color: #0f172a; font-size: 14px;")
            
            info_form.addRow(label, value)
        
        info_card.add_layout(info_form)
        layout.addWidget(info_card)
        
        # Close button
        close_btn = QPushButton("Đóng")
        close_btn.setMinimumHeight(44)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
        
        self.setLayout(layout)


class ModernAdminPanel(QMainWindow):
    """Modern Admin Panel - Full screen với Sidebar Navigation"""
    
    logout_signal = pyqtSignal()
    
    def __init__(self, user_info: dict, db, parent=None):
        super().__init__(parent)
        self.user_info = user_info
        self.db = db
        
        self.setWindowTitle("Admin Panel - Face Recognition System")
        
        # Models - OpenVINO components
        self.core = None
        self.face_detector = None
        self.landmarks_detector = None
        self.face_identifier = None
        self.faces_database = None
        self.models_loaded = False
        
        # Model paths - Load from database
        model_config = self.db.get_model_config()
        self.model_fd_path = model_config.get('model_fd_path', '')
        self.model_lm_path = model_config.get('model_lm_path', '')
        self.model_reid_path = model_config.get('model_reid_path', '')
        self.gallery_path = model_config.get('gallery_path', './gallery')
        
        # Pagination
        self.events_page = 1
        self.events_per_page = 20
        self.crops_page = 1
        self.crops_per_page = 20
        
        # Sidebar buttons list
        self.sidebar_buttons = []
        
        # Init UI
        self.init_ui()
        
        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_current_page)
        self.refresh_timer.setInterval(5000)
        self.refresh_timer.start()
        
        # Initial load
        self.navigate_to(0)
    
    def init_ui(self):
        """Initialize UI with sidebar"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header bar
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Content area with sidebar
        content_area = QWidget()
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Sidebar
        sidebar = self.create_sidebar()
        content_layout.addWidget(sidebar)
        
        # Main content area
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: #f8fafc;")
        
        content_inner_layout = QVBoxLayout()
        content_inner_layout.setContentsMargins(24, 24, 24, 24)
        
        # Stacked widget for different pages
        self.stacked_widget = QStackedWidget()
        
        # Add pages
        self.stacked_widget.addWidget(self.create_dashboard_page())      # 0
        self.stacked_widget.addWidget(self.create_events_page())         # 1
        self.stacked_widget.addWidget(self.create_crops_page())          # 2
        self.stacked_widget.addWidget(self.create_user_management_page())     # 3
        self.stacked_widget.addWidget(self.create_customer_management_page()) # 4
        self.stacked_widget.addWidget(self.create_model_config_page())   # 5
        self.stacked_widget.addWidget(self.create_system_settings_page())# 6
        
        content_inner_layout.addWidget(self.stacked_widget)
        self.content_widget.setLayout(content_inner_layout)
        
        content_layout.addWidget(self.content_widget, stretch=1)
        content_area.setLayout(content_layout)
        
        main_layout.addWidget(content_area)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: white;
                color: #64748b;
                border-top: 1px solid #e2e8f0;
                padding: 8px;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Chế độ Admin | {self.user_info['username']}")
        
        central.setLayout(main_layout)
    
    def create_header(self):
        """Create header bar"""
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #8b5cf6);
                border: none;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(24, 0, 24, 0)
        
        # Logo + Title
        logo_layout = QHBoxLayout()
        logo_layout.setSpacing(16)
        
        logo = QLabel("👁️")
        logo.setStyleSheet("font-size: 36px; border: none; color: white;")
        logo_layout.addWidget(logo)
        
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        title = QLabel("Face Recognition System")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: white; border: none;")
        title_layout.addWidget(title)
        
        subtitle = QLabel("Admin Panel")
        subtitle.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 12px; border: none;")
        title_layout.addWidget(subtitle)
        
        logo_layout.addLayout(title_layout)
        layout.addLayout(logo_layout)
        
        layout.addStretch()
        
        # User info + Logout
        user_layout = QHBoxLayout()
        user_layout.setSpacing(16)
        
        user_info = QLabel(f"👤 {self.user_info['username']}")
        user_info.setStyleSheet("color: white; font-size: 14px; font-weight: 600; border: none;")
        user_layout.addWidget(user_info)
        
        logout_btn = QPushButton("🚪 Đăng xuất")
        logout_btn.setCursor(Qt.PointingHandCursor)
        logout_btn.setFixedHeight(40)
        logout_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 8px;
                padding: 0 20px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.3);
            }
        """)
        logout_btn.clicked.connect(self.logout)
        user_layout.addWidget(logout_btn)
        
        layout.addLayout(user_layout)
        
        header.setLayout(layout)
        return header
    
    def create_sidebar(self):
        """Create sidebar navigation"""
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: white;
                border-right: 1px solid #e2e8f0;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(8)
        
        # Navigation title
        nav_title = QLabel("ĐIỀU HƯỚNG")
        nav_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 700; padding: 8px 12px;")
        layout.addWidget(nav_title)
        
        # Navigation buttons
        nav_items = [
            ("📊", "Dashboard"),
            ("📋", "Events"),
            ("🖼️", "Crops"),
            ("👥", "Quản lý Người dùng"),
            ("🧑", "Quản lý Khách hàng"),
            ("🤖", "Cấu hình Models"),
            ("⚙️", "Cài đặt Hệ thống"),
        ]
        
        for i, (icon, text) in enumerate(nav_items):
            btn = SidebarButton(icon, text)
            btn.clicked.connect(lambda checked, idx=i: self.navigate_to(idx))
            self.sidebar_buttons.append(btn)
            layout.addWidget(btn)
        
        layout.addStretch()
        
        # Version info
        version = QLabel("Version 2.0.0")
        version.setStyleSheet("color: #cbd5e1; font-size: 11px; padding: 8px 12px;")
        layout.addWidget(version)
        
        sidebar.setLayout(layout)
        return sidebar
    
    def navigate_to(self, index):
        """Navigate to specific page"""
        # Update sidebar buttons
        for i, btn in enumerate(self.sidebar_buttons):
            btn.set_active(i == index)
        
        # Switch page
        self.stacked_widget.setCurrentIndex(index)
        
        # Refresh content
        if index == 0:
            self.refresh_dashboard()
        elif index == 1:
            self.refresh_events()
        elif index == 2:
            self.refresh_crops()
    
    def create_dashboard_page(self):
        """Create dashboard page"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(24)
        
        # Page title
        title = QLabel("📊 Dashboard")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #0f172a;")
        layout.addWidget(title)
        
        # Stats cards row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)
        
        self.total_customers_card = StatsCard("👥", "Tổng Khách hàng", "0", "#6366f1")
        stats_layout.addWidget(self.total_customers_card)
        
        self.total_events_card = StatsCard("📋", "Tổng Events", "0", "#8b5cf6")
        stats_layout.addWidget(self.total_events_card)
        
        self.total_crops_card = StatsCard("🖼️", "Tổng Crops", "0", "#ec4899")
        stats_layout.addWidget(self.total_crops_card)
        
        self.vip_customers_card = StatsCard("⭐", "Khách VIP", "0", "#10b981")
        stats_layout.addWidget(self.vip_customers_card)
        
        layout.addLayout(stats_layout)
        
        # Recent activity card
        activity_card = ModernCard("📊 Hoạt động gần đây")
        
        self.activity_table = QTableWidget()
        self.activity_table.setColumnCount(4)
        self.activity_table.setHorizontalHeaderLabels(["Khách hàng", "Loại", "Thời gian", "Độ tin cậy"])
        self.activity_table.horizontalHeader().setStretchLastSection(True)
        self.activity_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.activity_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.activity_table.setAlternatingRowColors(True)
        self.activity_table.setMaximumHeight(300)
        self.activity_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
                gridline-color: #e2e8f0;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                color: #475569;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: 600;
            }
        """)
        
        activity_card.add_widget(self.activity_table)
        layout.addWidget(activity_card)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_events_page(self):
        """Create events page"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        # Page title
        title = QLabel("📋 Quản lý Events")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #0f172a;")
        layout.addWidget(title)
        
        # Search + Filter card
        search_card = ModernCard()
        search_layout = QHBoxLayout()
        search_layout.setSpacing(12)
        
        # Search box
        self.events_search = QLineEdit()
        self.events_search.setPlaceholderText("🔍 Tìm kiếm theo tên khách hàng...")
        self.events_search.setMinimumHeight(40)
        self.events_search.setStyleSheet("""
            QLineEdit {
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                padding: 0 16px;
                font-size: 14px;
                background-color: #f8fafc;
            }
            QLineEdit:focus {
                border-color: #6366f1;
                background-color: white;
            }
        """)
        self.events_search.textChanged.connect(self.filter_events)
        search_layout.addWidget(self.events_search, stretch=3)
        
        # Type filter
        self.event_type_filter = QComboBox()
        self.event_type_filter.addItems(["Tất cả", "ENTRY", "EXIT", "RECOGNIZED"])
        self.event_type_filter.setMinimumHeight(40)
        self.event_type_filter.setStyleSheet("""
            QComboBox {
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                padding: 0 16px;
                font-size: 14px;
                background-color: #f8fafc;
            }
            QComboBox:focus {
                border-color: #6366f1;
            }
        """)
        self.event_type_filter.currentTextChanged.connect(self.filter_events)
        search_layout.addWidget(self.event_type_filter, stretch=1)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Làm mới")
        refresh_btn.setMinimumHeight(40)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_events)
        search_layout.addWidget(refresh_btn)
        
        search_card.add_layout(search_layout)
        layout.addWidget(search_card)
        
        # Table card
        table_card = ModernCard()
        
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(6)
        self.events_table.setHorizontalHeaderLabels([
            "ID", "Khách hàng", "Loại", "Camera", "Độ tin cậy", "Thời gian"
        ])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.events_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.events_table.setAlternatingRowColors(True)
        self.events_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
                gridline-color: #e2e8f0;
            }
            QTableWidget::item {
                padding: 12px;
            }
            QTableWidget::item:selected {
                background-color: #e0e7ff;
                color: #1e293b;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                color: #475569;
                padding: 12px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: 600;
            }
        """)
        self.events_table.doubleClicked.connect(self.show_event_detail)
        
        table_card.add_widget(self.events_table)
        layout.addWidget(table_card)
        
        # Pagination
        pagination_layout = QHBoxLayout()
        pagination_layout.addStretch()
        
        prev_btn = QPushButton("← Trước")
        prev_btn.setMinimumHeight(36)
        prev_btn.setCursor(Qt.PointingHandCursor)
        prev_btn.clicked.connect(self.events_prev_page)
        prev_btn.setStyleSheet(self.pagination_btn_style())
        pagination_layout.addWidget(prev_btn)
        
        self.events_page_label = QLabel("Trang 1")
        self.events_page_label.setStyleSheet("padding: 0 16px; font-weight: 600; color: #475569;")
        pagination_layout.addWidget(self.events_page_label)
        
        next_btn = QPushButton("Sau →")
        next_btn.setMinimumHeight(36)
        next_btn.setCursor(Qt.PointingHandCursor)
        next_btn.clicked.connect(self.events_next_page)
        next_btn.setStyleSheet(self.pagination_btn_style())
        pagination_layout.addWidget(next_btn)
        
        pagination_layout.addStretch()
        layout.addLayout(pagination_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_crops_page(self):
        """Create crops page"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        # Page title
        title = QLabel("🖼️ Quản lý Crops")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #0f172a;")
        layout.addWidget(title)
        
        # Search + Filter bar
        search_card = ModernCard()
        search_layout = QHBoxLayout()
        search_layout.setSpacing(12)
        
        # Search box
        self.crops_search = QLineEdit()
        self.crops_search.setPlaceholderText("🔍 Tìm kiếm crops theo tên khách hàng...")
        self.crops_search.setMinimumHeight(40)
        self.crops_search.setStyleSheet("""
            QLineEdit {
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                padding: 0 16px;
                font-size: 14px;
                background-color: #f8fafc;
            }
            QLineEdit:focus {
                border-color: #6366f1;
                background-color: white;
            }
        """)
        self.crops_search.textChanged.connect(self.filter_crops)
        search_layout.addWidget(self.crops_search, stretch=2)
        
        # Items per page selector
        per_page_label = QLabel("Hiển thị:")
        per_page_label.setStyleSheet("color: #475569; font-weight: 600;")
        search_layout.addWidget(per_page_label)
        
        self.crops_per_page_combo = QComboBox()
        self.crops_per_page_combo.addItems(["10", "20", "30", "50"])
        self.crops_per_page_combo.setCurrentText("20")
        self.crops_per_page_combo.setMinimumHeight(40)
        self.crops_per_page_combo.setStyleSheet("""
            QComboBox {
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                padding: 0 16px;
                font-size: 14px;
                background-color: #f8fafc;
                min-width: 80px;
            }
            QComboBox:focus {
                border-color: #6366f1;
            }
        """)
        self.crops_per_page_combo.currentTextChanged.connect(self.change_crops_per_page)
        search_layout.addWidget(self.crops_per_page_combo)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Làm mới")
        refresh_btn.setMinimumHeight(40)
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_crops)
        search_layout.addWidget(refresh_btn)
        
        search_card.add_layout(search_layout)
        layout.addWidget(search_card)
        
        # Table card
        table_card = ModernCard()
        
        self.crops_table = QTableWidget()
        self.crops_table.setColumnCount(6)
        self.crops_table.setHorizontalHeaderLabels([
            "ID", "Thumbnail", "Khách hàng", "Độ tin cậy", "Event Type", "Thời gian"
        ])
        
        # Configure columns
        self.crops_table.horizontalHeader().setStretchLastSection(False)
        self.crops_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.crops_table.setColumnWidth(0, 60)
        self.crops_table.setColumnWidth(1, 100)
        self.crops_table.setColumnWidth(3, 100)
        self.crops_table.setColumnWidth(4, 120)
        self.crops_table.setColumnWidth(5, 160)
        
        self.crops_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.crops_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.crops_table.setAlternatingRowColors(True)
        self.crops_table.verticalHeader().setDefaultSectionSize(90)
        
        self.crops_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
                gridline-color: #e2e8f0;
            }
            QTableWidget::item {
                padding: 12px;
            }
            QTableWidget::item:selected {
                background-color: #e0e7ff;
                color: #1e293b;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                color: #475569;
                padding: 12px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: 600;
            }
        """)
        self.crops_table.doubleClicked.connect(self.show_crop_detail_from_table)
        
        table_card.add_widget(self.crops_table)
        layout.addWidget(table_card)
        
        # Pagination
        pagination_layout = QHBoxLayout()
        pagination_layout.addStretch()
        
        prev_btn = QPushButton("← Trước")
        prev_btn.setMinimumHeight(36)
        prev_btn.setCursor(Qt.PointingHandCursor)
        prev_btn.clicked.connect(self.crops_prev_page)
        prev_btn.setStyleSheet(self.pagination_btn_style())
        pagination_layout.addWidget(prev_btn)
        
        self.crops_page_label = QLabel("Trang 1")
        self.crops_page_label.setStyleSheet("padding: 0 16px; font-weight: 600; color: #475569;")
        pagination_layout.addWidget(self.crops_page_label)
        
        next_btn = QPushButton("Sau →")
        next_btn.setMinimumHeight(36)
        next_btn.setCursor(Qt.PointingHandCursor)
        next_btn.clicked.connect(self.crops_next_page)
        next_btn.setStyleSheet(self.pagination_btn_style())
        pagination_layout.addWidget(next_btn)
        
        pagination_layout.addStretch()
        layout.addLayout(pagination_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_user_management_page(self):
        """Create user management page"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(24)
        
        # Page title
        title = QLabel("👥 Quản lý Người dùng")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #0f172a;")
        layout.addWidget(title)
        
        # Action card
        action_card = ModernCard("Quản lý tài khoản người dùng hệ thống")
        
        desc = QLabel("Thêm, sửa, xóa tài khoản người dùng và phân quyền truy cập hệ thống.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748b; font-size: 14px; margin-bottom: 16px;")
        action_card.add_widget(desc)
        
        open_btn = QPushButton("🔓 Mở Quản lý Người dùng")
        open_btn.setMinimumHeight(50)
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                padding: 12px 32px;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        open_btn.clicked.connect(self.manage_users)
        action_card.add_widget(open_btn)
        
        layout.addWidget(action_card)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_customer_management_page(self):
        """Create customer management page"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(24)
        
        # Page title
        title = QLabel("🧑 Quản lý Khách hàng")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #0f172a;")
        layout.addWidget(title)
        
        # Action card
        action_card = ModernCard("Quản lý thông tin khách hàng VIP")
        
        desc = QLabel("Thêm, chỉnh sửa thông tin khách hàng, quản lý phân khúc khách hàng và cập nhật dữ liệu nhận diện khuôn mặt.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748b; font-size: 14px; margin-bottom: 16px;")
        action_card.add_widget(desc)
        
        open_btn = QPushButton("🔓 Mở Quản lý Khách hàng")
        open_btn.setMinimumHeight(50)
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                padding: 12px 32px;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        open_btn.clicked.connect(self.manage_customers)
        action_card.add_widget(open_btn)
        
        layout.addWidget(action_card)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_model_config_page(self):
        """Create model configuration page"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(24)
        
        # Page title
        title = QLabel("🤖 Cấu hình Models AI")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #0f172a;")
        layout.addWidget(title)
        
        # Configuration card
        config_card = ModernCard("Cài đặt các mô hình AI")
        
        form = QFormLayout()
        form.setSpacing(16)
        
        # Face Detection
        self.fd_input = QLineEdit()
        self.fd_input.setPlaceholderText("Chọn file .xml cho Face Detection")
        self.fd_input.setMinimumHeight(40)
        fd_btn = QPushButton("📂 Browse...")
        fd_btn.setCursor(Qt.PointingHandCursor)
        fd_btn.clicked.connect(lambda: self.browse_file(self.fd_input))
        fd_layout = QHBoxLayout()
        fd_layout.addWidget(self.fd_input)
        fd_layout.addWidget(fd_btn)
        form.addRow("Face Detection:", fd_layout)
        
        # Landmarks
        self.lm_input = QLineEdit()
        self.lm_input.setPlaceholderText("Chọn file .xml cho Landmarks")
        self.lm_input.setMinimumHeight(40)
        lm_btn = QPushButton("📂 Browse...")
        lm_btn.setCursor(Qt.PointingHandCursor)
        lm_btn.clicked.connect(lambda: self.browse_file(self.lm_input))
        lm_layout = QHBoxLayout()
        lm_layout.addWidget(self.lm_input)
        lm_layout.addWidget(lm_btn)
        form.addRow("Landmarks:", lm_layout)
        
        # Re-ID
        self.reid_input = QLineEdit()
        self.reid_input.setPlaceholderText("Chọn file .xml cho Re-ID")
        self.reid_input.setMinimumHeight(40)
        reid_btn = QPushButton("📂 Browse...")
        reid_btn.setCursor(Qt.PointingHandCursor)
        reid_btn.clicked.connect(lambda: self.browse_file(self.reid_input))
        reid_layout = QHBoxLayout()
        reid_layout.addWidget(self.reid_input)
        reid_layout.addWidget(reid_btn)
        form.addRow("Re-ID:", reid_layout)
        
        # Gallery
        self.gallery_input = QLineEdit()
        self.gallery_input.setPlaceholderText("Chọn thư mục Gallery")
        self.gallery_input.setMinimumHeight(40)
        gallery_btn = QPushButton("📂 Browse...")
        gallery_btn.setCursor(Qt.PointingHandCursor)
        gallery_btn.clicked.connect(lambda: self.browse_directory(self.gallery_input))
        gallery_layout = QHBoxLayout()
        gallery_layout.addWidget(self.gallery_input)
        gallery_layout.addWidget(gallery_btn)
        form.addRow("Gallery:", gallery_layout)
        
        config_card.add_layout(form)
        
        # Save button
        save_btn = QPushButton("💾 Lưu Cấu hình")
        save_btn.setMinimumHeight(50)
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                padding: 12px 32px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        save_btn.clicked.connect(self.save_model_config)
        config_card.add_widget(save_btn)
        
        layout.addWidget(config_card)
        
        # Load current config
        self.load_model_config()
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_system_settings_page(self):
        """Create system settings page"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(24)
        
        # Page title
        title = QLabel("⚙️ Cài đặt Hệ thống")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #0f172a;")
        layout.addWidget(title)
        
        # Settings card
        settings_card = ModernCard("Cấu hình thông số hệ thống")
        
        form = QFormLayout()
        form.setSpacing(20)
        
        # Detection Cooldown
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setMinimum(1.0)
        self.cooldown_spin.setMaximum(60.0)
        self.cooldown_spin.setSingleStep(0.5)
        self.cooldown_spin.setValue(self.db.get_detection_cooldown())
        self.cooldown_spin.setSuffix(" giây")
        self.cooldown_spin.setMinimumHeight(40)
        form.addRow("Thời gian Cooldown:", self.cooldown_spin)
        
        # Revisit Threshold
        self.revisit_spin = QDoubleSpinBox()
        self.revisit_spin.setMinimum(0.5)
        self.revisit_spin.setMaximum(24.0)
        self.revisit_spin.setSingleStep(0.5)
        self.revisit_spin.setValue(self.db.get_revisit_threshold())
        self.revisit_spin.setSuffix(" giờ")
        self.revisit_spin.setMinimumHeight(40)
        form.addRow("Thời gian Revisit:", self.revisit_spin)
        
        # Capture Interval
        self.capture_spin = QDoubleSpinBox()
        self.capture_spin.setMinimum(1.0)
        self.capture_spin.setMaximum(10.0)
        self.capture_spin.setSingleStep(0.5)
        self.capture_spin.setValue(self.db.get_capture_interval())
        self.capture_spin.setSuffix(" giây")
        self.capture_spin.setMinimumHeight(40)
        form.addRow("Thời gian chụp:", self.capture_spin)
        
        settings_card.add_layout(form)
        
        # Info box
        info = QLabel(
            "💡 Hướng dẫn:\n\n"
            "• Cooldown: Thời gian tối thiểu giữa 2 lần phát hiện (3-10s)\n"
            "• Revisit: Thời gian để tính lượt ghé mới (2-6h)\n"
            "• Thời gian chụp: Khoảng cách giữa các lần chụp ảnh (1-5s)"
        )
        info.setWordWrap(True)
        info.setStyleSheet("""
            color: #64748b;
            background-color: #f8fafc;
            padding: 16px;
            border-radius: 8px;
            border: none;
            margin-top: 12px;
        """)
        settings_card.add_widget(info)
        
        # Save button
        save_btn = QPushButton("💾 Lưu Cài đặt")
        save_btn.setMinimumHeight(50)
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                padding: 12px 32px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        save_btn.clicked.connect(self.save_system_settings)
        settings_card.add_widget(save_btn)
        
        layout.addWidget(settings_card)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def pagination_btn_style(self):
        """Style for pagination buttons"""
        return """
            QPushButton {
                background-color: white;
                color: #475569;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                border-color: #6366f1;
                color: #6366f1;
            }
        """
    
    # Dashboard methods
    def refresh_dashboard(self):
        """Refresh dashboard statistics"""
        try:
            # Get statistics
            total_customers = len(self.db.get_all_customers())
            total_events = len(self.db.get_recent_events(limit=1000))
            total_crops = len(self.db.get_recent_crops(limit=1000))
            vip_customers = len([c for c in self.db.get_all_customers() if c.segment == CustomerSegment.VIP])
            
            # Update stats cards
            self.total_customers_card.set_value(str(total_customers))
            self.total_events_card.set_value(str(total_events))
            self.total_crops_card.set_value(str(total_crops))
            self.vip_customers_card.set_value(str(vip_customers))
            
            # Update recent activity table
            recent_events = self.db.get_recent_events(limit=10)
            self.activity_table.setRowCount(len(recent_events))
            
            for row, event in enumerate(recent_events):
                self.activity_table.setItem(row, 0, QTableWidgetItem(event.customer_name))
                
                # Hiển thị event type theo segment (đã được set trong events_manager)
                event_type_display = event.event_type
                # Format event type để dễ đọc hơn
                if event_type_display == EventType.VIP_DETECTED:
                    event_type_display = "VIP"
                elif event_type_display == EventType.NEW_CUSTOMER:
                    event_type_display = "Khách mới"
                elif event_type_display == EventType.REGULAR_VISIT:
                    event_type_display = "Khách thường"
                elif event_type_display == EventType.BLACKLIST:
                    event_type_display = "Blacklist"
                
                self.activity_table.setItem(row, 1, QTableWidgetItem(event_type_display))
                
                timestamp = event.timestamp.strftime("%H:%M:%S") if event.timestamp else "N/A"
                self.activity_table.setItem(row, 2, QTableWidgetItem(timestamp))
                
                # Hiển thị confidence summary từ metadata nếu có, nếu không dùng confidence đầu tiên
                confidence_display = f"{event.confidence:.1f}%"
                if event.metadata and isinstance(event.metadata, dict):
                    if 'confidence_avg' in event.metadata:
                        confidence_display = f"{event.metadata['confidence_avg']:.1f}%"
                    elif 'confidences' in event.metadata and len(event.metadata['confidences']) > 0:
                        avg_conf = sum(event.metadata['confidences']) / len(event.metadata['confidences'])
                        confidence_display = f"{avg_conf:.1f}%"
                
                self.activity_table.setItem(row, 3, QTableWidgetItem(confidence_display))
            
        except Exception as e:
            log.error(f"Error refreshing dashboard: {e}")
    
    # Events methods
    def refresh_events(self):
        """Refresh events table"""
        try:
            events = self.db.get_recent_events(limit=self.events_per_page * 10)
            
            # Apply filters
            search_text = self.events_search.text().lower()
            event_type = self.event_type_filter.currentText()
            
            filtered_events = []
            for event in events:
                if search_text and search_text not in event.customer_name.lower():
                    continue
                if event_type != "Tất cả" and event.event_type != event_type:
                    continue
                filtered_events.append(event)
            
            # Pagination
            start_idx = (self.events_page - 1) * self.events_per_page
            end_idx = start_idx + self.events_per_page
            page_events = filtered_events[start_idx:end_idx]
            
            # Update table
            self.events_table.setRowCount(len(page_events))
            
            for row, event in enumerate(page_events):
                self.events_table.setItem(row, 0, QTableWidgetItem(str(event.id)))
                self.events_table.setItem(row, 1, QTableWidgetItem(event.customer_name))
                
                # Format event type theo segment
                event_type_display = event.event_type
                if event.event_type == EventType.VIP_DETECTED:
                    event_type_display = "VIP"
                elif event.event_type == EventType.NEW_CUSTOMER:
                    event_type_display = "Khách mới"
                elif event.event_type == EventType.REGULAR_VISIT:
                    event_type_display = "Khách thường"
                elif event.event_type == EventType.BLACKLIST:
                    event_type_display = "Blacklist"
                elif event.event_type == EventType.UNKNOWN:
                    event_type_display = "Unknown"
                
                self.events_table.setItem(row, 2, QTableWidgetItem(event_type_display))
                
                camera = f"Camera #{event.camera_id}" if event.camera_id else "N/A"
                self.events_table.setItem(row, 3, QTableWidgetItem(camera))
                
                # Hiển thị confidence summary từ metadata nếu có
                confidence_display = f"{event.confidence:.1f}%"
                if event.metadata and isinstance(event.metadata, dict):
                    if 'confidence_avg' in event.metadata:
                        confidence_display = f"{event.metadata['confidence_avg']:.1f}%"
                    elif 'confidences' in event.metadata and len(event.metadata['confidences']) > 0:
                        avg_conf = sum(event.metadata['confidences']) / len(event.metadata['confidences'])
                        confidence_display = f"{avg_conf:.1f}%"
                
                self.events_table.setItem(row, 4, QTableWidgetItem(confidence_display))
                
                timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S") if event.timestamp else "N/A"
                self.events_table.setItem(row, 5, QTableWidgetItem(timestamp))
            
            # Update page label
            total_pages = (len(filtered_events) + self.events_per_page - 1) // self.events_per_page
            self.events_page_label.setText(f"Trang {self.events_page}/{max(1, total_pages)}")
            
        except Exception as e:
            log.error(f"Error refreshing events: {e}")
    
    def filter_events(self):
        """Filter events"""
        self.events_page = 1
        self.refresh_events()
    
    def events_prev_page(self):
        """Previous page"""
        if self.events_page > 1:
            self.events_page -= 1
            self.refresh_events()
    
    def events_next_page(self):
        """Next page"""
        self.events_page += 1
        self.refresh_events()
    
    def show_event_detail(self, index):
        """Show event detail dialog"""
        row = index.row()
        event_id = int(self.events_table.item(row, 0).text())
        
        event = None
        for e in self.db.get_recent_events(limit=1000):
            if e.id == event_id:
                event = e
                break
        
        if event:
            dialog = EventDetailDialog(event, self.db, self)
            dialog.exec_()
    
    # Crops methods
    def create_thumbnail_widget(self, crop):
        """Create thumbnail widget for table cell"""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setAlignment(Qt.AlignCenter)
        
        image_label = QLabel()
        image_label.setFixedSize(80, 80)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #e2e8f0;
                border-radius: 4px;
                background-color: #f8fafc;
            }
        """)
        
        try:
            file_path = Path(crop.file_path)
            if file_path.exists():
                pixmap = QPixmap(str(file_path))
                if not pixmap.isNull():
                    scaled = pixmap.scaled(76, 76, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    image_label.setPixmap(scaled)
                else:
                    image_label.setText("N/A")
            else:
                image_label.setText("N/A")
        except Exception as e:
            log.error(f"Error loading thumbnail: {e}")
            image_label.setText("Error")
        
        layout.addWidget(image_label)
        container.setLayout(layout)
        return container
    
    def refresh_crops(self):
        """Refresh crops table"""
        try:
            crops = self.db.get_recent_crops(limit=1000)
            
            # Apply filter
            search_text = self.crops_search.text().lower()
            filtered_crops = []
            
            for crop in crops:
                customer = self.db.get_customer(crop.customer_id) if crop.customer_id else None
                customer_name = customer.name if customer else "Unknown"
                
                if search_text and search_text not in customer_name.lower():
                    continue
                
                filtered_crops.append(crop)
            
            # Pagination
            start_idx = (self.crops_page - 1) * self.crops_per_page
            end_idx = start_idx + self.crops_per_page
            page_crops = filtered_crops[start_idx:end_idx]
            
            # Update table
            self.crops_table.setRowCount(len(page_crops))
            
            for row, crop in enumerate(page_crops):
                # ID
                self.crops_table.setItem(row, 0, QTableWidgetItem(str(crop.id)))
                
                # Thumbnail
                thumbnail_widget = self.create_thumbnail_widget(crop)
                self.crops_table.setCellWidget(row, 1, thumbnail_widget)
                
                # Customer name
                customer = self.db.get_customer(crop.customer_id) if crop.customer_id else None
                customer_name = customer.name if customer else "Unknown"
                self.crops_table.setItem(row, 2, QTableWidgetItem(customer_name))
                
                # Confidence
                self.crops_table.setItem(row, 3, QTableWidgetItem(f"{crop.confidence:.1f}%"))
                
                # Event Type
                event_type = "N/A"
                if hasattr(crop, 'event_id') and crop.event_id:
                    try:
                        events = self.db.get_recent_events(limit=1000)
                        event = next((e for e in events if e.id == crop.event_id), None)
                        if event:
                            event_type = event.event_type
                    except Exception as e:
                        log.error(f"Error getting event for crop: {e}")
                
                self.crops_table.setItem(row, 4, QTableWidgetItem(event_type))
                
                # Timestamp
                if hasattr(crop, 'timestamp') and crop.timestamp:
                    timestamp = crop.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    timestamp = "N/A"
                self.crops_table.setItem(row, 5, QTableWidgetItem(timestamp))
            
            # Update page label
            total_crops = len(filtered_crops)
            total_pages = (total_crops + self.crops_per_page - 1) // self.crops_per_page
            self.crops_page_label.setText(f"Trang {self.crops_page}/{max(1, total_pages)}")
            
        except Exception as e:
            log.error(f"Error refreshing crops: {e}")
    
    def change_crops_per_page(self, value):
        """Change items per page"""
        self.crops_per_page = int(value)
        self.crops_page = 1
        self.refresh_crops()
    
    def show_crop_detail_from_table(self, index):
        """Show crop detail from table row"""
        try:
            row = index.row()
            crop_id = int(self.crops_table.item(row, 0).text())
            
            # Find crop from database
            crops = self.db.get_recent_crops(limit=1000)
            crop = next((c for c in crops if c.id == crop_id), None)
            
            if crop:
                self.show_crop_detail(crop)
            else:
                QMessageBox.warning(self, "Lỗi", "Không tìm thấy crop")
        except Exception as e:
            log.error(f"Error showing crop detail from table: {e}")
            QMessageBox.warning(self, "Lỗi", f"Không thể hiển thị chi tiết: {e}")
    
    def show_crop_detail(self, crop):
        """Show crop detail dialog"""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Chi tiết Crop #{crop.id}")
            dialog.setMinimumSize(600, 700)
            dialog.setStyleSheet("background-color: #f8fafc;")
            
            layout = QVBoxLayout()
            layout.setSpacing(20)
            layout.setContentsMargins(24, 24, 24, 24)
            
            # Header
            header = QLabel(f"Crop #{crop.id}")
            header_font = QFont()
            header_font.setPointSize(18)
            header_font.setBold(True)
            header.setFont(header_font)
            header.setStyleSheet("color: #0f172a;")
            layout.addWidget(header)
            
            # Image card
            image_card = ModernCard()
            file_path = Path(crop.file_path)
            if file_path.exists():
                pixmap = QPixmap(str(file_path))
                if not pixmap.isNull():
                    image_label = QLabel()
                    scaled = pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    image_label.setPixmap(scaled)
                    image_label.setAlignment(Qt.AlignCenter)
                    image_card.add_widget(image_label)
            
            layout.addWidget(image_card)
            
            # Info card
            info_card = ModernCard()
            info_form = QFormLayout()
            info_form.setSpacing(12)
            
            customer = self.db.get_customer(crop.customer_id) if crop.customer_id else None
            customer_name = customer.name if customer else "Unknown"
            
            details = [
                ("Khách hàng:", customer_name),
                ("Độ tin cậy:", f"{crop.confidence:.1f}%"),
                ("File path:", crop.file_path),
                ("Crop ID:", str(crop.id)),
            ]
            
            if hasattr(crop, 'timestamp') and crop.timestamp:
                details.append(("Thời gian:", crop.timestamp.strftime("%Y-%m-%d %H:%M:%S")))
            
            for label_text, value_text in details:
                label = QLabel(label_text)
                label.setStyleSheet("color: #64748b; font-weight: 600; font-size: 13px;")
                
                value = QLabel(str(value_text))
                value.setStyleSheet("color: #0f172a; font-size: 14px;")
                value.setWordWrap(True)
                
                info_form.addRow(label, value)
            
            info_card.add_layout(info_form)
            layout.addWidget(info_card)
            
            # Close button
            close_btn = QPushButton("Đóng")
            close_btn.setMinimumHeight(44)
            close_btn.setCursor(Qt.PointingHandCursor)
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6366f1;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                    padding: 12px 24px;
                }
                QPushButton:hover {
                    background-color: #4f46e5;
                }
            """)
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn, alignment=Qt.AlignRight)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            log.error(f"Error showing crop detail: {e}")
            QMessageBox.warning(self, "Lỗi", f"Không thể hiển thị chi tiết: {e}")
    
    def filter_crops(self):
        """Filter crops"""
        self.crops_page = 1
        self.refresh_crops()
    
    def crops_prev_page(self):
        """Previous page"""
        if self.crops_page > 1:
            self.crops_page -= 1
            self.refresh_crops()
    
    def crops_next_page(self):
        """Next page"""
        self.crops_page += 1
        self.refresh_crops()
    
    # Management methods
    def manage_users(self):
        """Open user management dialog"""
        try:
            dialog = UserManagementDialog(self.db, self)
            dialog.exec_()
        except Exception as e:
            log.error(f"Error opening user management: {e}")
            QMessageBox.warning(self, "Lỗi", f"Không thể mở quản lý người dùng: {e}")
    
    def load_models(self):
        """Load face recognition models from database configuration"""
        # Reset models first to ensure clean state
        self.face_detector = None
        self.landmarks_detector = None
        self.face_identifier = None
        self.faces_database = None
        self.models_loaded = False
        
        try:
            # Reload model config from database (in case it was updated)
            model_config = self.db.get_model_config()
            self.model_fd_path = model_config.get('model_fd_path', '')
            self.model_lm_path = model_config.get('model_lm_path', '')
            self.model_reid_path = model_config.get('model_reid_path', '')
            self.gallery_path = model_config.get('gallery_path', './gallery')
            
            log.info(f"Loading models from config: FD={self.model_fd_path}, LM={self.model_lm_path}, ReID={self.model_reid_path}")
            
            # Check if Core and utils are available
            if Core is None or FaceDetector is None:
                log.error("OpenVINO or utils modules not available")
                return False
            
            # Check if model files exist
            if not all([
                Path(self.model_fd_path).exists() if self.model_fd_path else False,
                Path(self.model_lm_path).exists() if self.model_lm_path else False,
                Path(self.model_reid_path).exists() if self.model_reid_path else False
            ]):
                log.warning(f"Model files not found. FD={self.model_fd_path}, LM={self.model_lm_path}, ReID={self.model_reid_path}")
                return False
            
            # Initialize OpenVINO Core
            self.core = Core()
            log.info(f"OpenVINO Version: {get_version()}")
            
            # Load Face Detector
            log.info(f"Loading Face Detection model: {self.model_fd_path}")
            try:
                self.face_detector = FaceDetector(
                    self.core,
                    Path(self.model_fd_path),
                    input_size=(0, 0),
                    confidence_threshold=0.6,
                    roi_scale_factor=1.15
                )
                self.face_detector.deploy("CPU")
                log.info("✓ Face Detection model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load Face Detection model: {e}", exc_info=True)
                raise
            
            # Load Landmarks Detector
            log.info(f"Loading Landmarks model: {self.model_lm_path}")
            try:
                self.landmarks_detector = LandmarksDetector(
                    self.core,
                    Path(self.model_lm_path)
                )
                self.landmarks_detector.deploy("CPU", 16)
                log.info("✓ Landmarks Detection model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load Landmarks Detection model: {e}", exc_info=True)
                raise
            
            # Load Face Identifier
            log.info(f"Loading Re-ID model: {self.model_reid_path}")
            try:
                self.face_identifier = FaceIdentifier(
                    self.core,
                    Path(self.model_reid_path),
                    match_threshold=0.3,
                    match_algo='HUNGARIAN'
                )
                self.face_identifier.deploy("CPU", 16)
                log.info("✓ Face Reidentification model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load Face Reidentification model: {e}", exc_info=True)
                raise
            
            # Ensure gallery folder exists before loading Faces Database
            gallery_path_obj = Path(self.gallery_path)
            if not gallery_path_obj.exists():
                log.info(f"Gallery folder does not exist, creating: {self.gallery_path}")
                gallery_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Load Faces Database
            log.info(f"Building faces database from: {self.gallery_path}")
            try:
                self.faces_database = FacesDatabase(
                    self.gallery_path,
                    self.face_identifier,
                    self.landmarks_detector,
                    self.face_detector,
                    no_show=True
                )
                self.face_identifier.set_faces_database(self.faces_database)
                log.info("✓ Faces database built successfully")
            except Exception as e:
                log.error(f"Error initializing FacesDatabase: {e}", exc_info=True)
                raise  # Re-raise to be caught by outer exception handler
            
            # Verify all models are loaded correctly
            models_status = {
                'face_detector': self.face_detector is not None,
                'landmarks_detector': self.landmarks_detector is not None,
                'face_identifier': self.face_identifier is not None,
                'faces_database': self.faces_database is not None
            }
            
            if not all(models_status.values()):
                failed_models = [name for name, status in models_status.items() if not status]
                log.error(f"Some models failed to load properly. Failed models: {failed_models}")
                log.error(f"Model status: {models_status}")
                self.face_detector = None
                self.landmarks_detector = None
                self.face_identifier = None
                self.faces_database = None
                self.models_loaded = False
                return False
            
            log.info("✓ All models verified successfully")
            
            self.models_loaded = True
            log.info(f"Models loaded successfully. Database: {len(self.faces_database)} identities")
            log.debug(f"Models verification: face_detector={self.face_detector is not None}, "
                     f"landmarks_detector={self.landmarks_detector is not None}, "
                     f"face_identifier={self.face_identifier is not None}, "
                     f"faces_database={self.faces_database is not None}")
            return True
            
        except Exception as e:
            log.error(f"Error loading models: {e}", exc_info=True)
            # Ensure models are reset on error
            self.face_detector = None
            self.landmarks_detector = None
            self.face_identifier = None
            self.faces_database = None
            self.models_loaded = False
            return False
    
    def manage_customers(self):
        """Open customer management dialog"""
        try:
            # Load models if not already loaded
            if not self.models_loaded:
                # Show loading message
                reply = QMessageBox.question(
                    self,
                    "Load Models",
                    "Models chưa được load. Bạn có muốn load models ngay bây giờ?\n\n"
                    "Lưu ý: Cần load models để thêm khách hàng với ảnh khuôn mặt.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    QMessageBox.information(
                        self,
                        "Đang load models",
                        "Đang tải models... Vui lòng đợi."
                    )
                    if not self.load_models():
                        QMessageBox.warning(
                            self,
                            "Lỗi",
                            "Không thể load models!\n\n"
                            "Vui lòng kiểm tra:\n"
                            "• Đường dẫn models trong Cấu hình Models AI\n"
                            "• Models có tồn tại không\n"
                            "• Đã cài đặt OpenVINO chưa"
                        )
                        return
                else:
                    # User chose not to load models
                    return
            
            # Verify models are actually loaded (not just flag set)
            # Use explicit None check because FacesDatabase can be empty (0 identities) but still valid
            models_loaded = (
                self.face_detector is not None and
                self.landmarks_detector is not None and
                self.face_identifier is not None and
                self.faces_database is not None
            )
            
            if not models_loaded:
                log.warning(f"Models flag is True but models are None. face_detector={self.face_detector is not None}, "
                           f"landmarks_detector={self.landmarks_detector is not None}, "
                           f"face_identifier={self.face_identifier is not None}, "
                           f"faces_database={self.faces_database is not None}")
                
                # Try to reload models
                QMessageBox.information(
                    self,
                    "Đang load models",
                    "Models chưa được load đầy đủ. Đang tải lại models... Vui lòng đợi."
                )
                if not self.load_models():
                    QMessageBox.warning(
                        self,
                        "Lỗi",
                        "Không thể load models!\n\n"
                        "Vui lòng kiểm tra:\n"
                        "• Đường dẫn models trong Cấu hình Models AI\n"
                        "• Models có tồn tại không\n"
                        "• Đã cài đặt OpenVINO chưa"
                    )
                    return
            
            # Open dialog with models
            log.info(f"Opening customer management dialog with models loaded: "
                    f"face_detector={self.face_detector is not None}, "
                    f"landmarks_detector={self.landmarks_detector is not None}, "
                    f"face_identifier={self.face_identifier is not None}, "
                    f"faces_database={self.faces_database is not None}")
            
            # Double-check models before passing to dialog
            # Use explicit None check instead of truthiness check
            # (faces_database can be empty but still valid)
            models_ok = (
                self.face_detector is not None and
                self.landmarks_detector is not None and
                self.face_identifier is not None and
                self.faces_database is not None
            )
            
            if not models_ok:
                log.error("Cannot open customer management dialog: models are not fully loaded")
                log.error(f"Model status: face_detector={self.face_detector is not None}, "
                         f"landmarks_detector={self.landmarks_detector is not None}, "
                         f"face_identifier={self.face_identifier is not None}, "
                         f"faces_database={self.faces_database is not None}")
                QMessageBox.warning(
                    self,
                    "Lỗi",
                    "Models chưa được load đầy đủ!\n\n"
                    "Vui lòng load models trước khi mở quản lý khách hàng."
                )
                return
            
            dialog = CustomerManagementDialog(
                self.db,
                self,
                face_detector=self.face_detector,
                landmarks_detector=self.landmarks_detector,
                face_identifier=self.face_identifier,
                faces_database=self.faces_database,
                gallery_path=self.gallery_path
            )
            dialog.exec_()
        except Exception as e:
            log.error(f"Error opening customer management: {e}", exc_info=True)
            QMessageBox.warning(self, "Lỗi", f"Không thể mở quản lý khách hàng: {e}")
    
    def load_model_config(self):
        """Load current model configuration"""
        try:
            config = self.db.get_model_config()
            self.fd_input.setText(config.get('model_fd_path', ''))
            self.lm_input.setText(config.get('model_lm_path', ''))
            self.reid_input.setText(config.get('model_reid_path', ''))
            self.gallery_input.setText(config.get('gallery_path', ''))
            
            # Update internal paths
            self.model_fd_path = config.get('model_fd_path', '')
            self.model_lm_path = config.get('model_lm_path', '')
            self.model_reid_path = config.get('model_reid_path', '')
            self.gallery_path = config.get('gallery_path', './gallery')
        except Exception as e:
            log.error(f"Error loading model config: {e}")
    
    def save_model_config(self):
        """Save model configuration"""
        try:
            # Save to database
            self.db.save_model_config(
                self.fd_input.text(),
                self.lm_input.text(),
                self.reid_input.text(),
                self.gallery_input.text()
            )
            
            # Update internal paths
            self.model_fd_path = self.fd_input.text()
            self.model_lm_path = self.lm_input.text()
            self.model_reid_path = self.reid_input.text()
            self.gallery_path = self.gallery_input.text()
            
            # Reset models_loaded flag and models since config changed
            # Models cũ không còn hợp lệ với config mới
            self.models_loaded = False
            self.face_detector = None
            self.landmarks_detector = None
            self.face_identifier = None
            self.faces_database = None
            
            # Ask if user wants to load models now
            reply = QMessageBox.question(
                self,
                "Load Models",
                "Cấu hình models đã được lưu!\n\n"
                "Bạn có muốn load models ngay bây giờ?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                if self.load_models():
                    QMessageBox.information(
                        self,
                        "Thành công",
                        f"Cấu hình đã được lưu và models đã được load!\n\n"
                        f"Database: {len(self.faces_database)} identities"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Cảnh báo",
                        "Cấu hình đã được lưu nhưng không thể load models!\n\n"
                        "Vui lòng kiểm tra:\n"
                        "• Đường dẫn models có đúng không\n"
                        "• Models có tồn tại không\n"
                        "• Đã cài đặt OpenVINO chưa"
                    )
            else:
                # User chose not to load models - show info that they need to load later
                QMessageBox.information(
                    self,
                    "Đã lưu",
                    "Cấu hình đã được lưu!\n\n"
                    "Vui lòng load models trước khi sử dụng các tính năng nhận diện."
                )
                
        except Exception as e:
            log.error(f"Error saving model config: {e}")
            QMessageBox.warning(self, "Lỗi", f"Không thể lưu cấu hình: {e}")
    
    def save_system_settings(self):
        """Save system settings"""
        try:
            self.db.set_detection_cooldown(self.cooldown_spin.value())
            self.db.set_revisit_threshold(self.revisit_spin.value())
            self.db.set_capture_interval(self.capture_spin.value())
            QMessageBox.information(self, "Thành công", "Cài đặt đã được cập nhật!")
        except Exception as e:
            log.error(f"Error saving system settings: {e}")
            QMessageBox.warning(self, "Lỗi", f"Không thể lưu cài đặt: {e}")
    
    def browse_file(self, line_edit):
        """Browse for file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file",
            "./models",
            "Model Files (*.xml *.onnx);;All Files (*)"
        )
        if file_path:
            line_edit.setText(file_path)
    
    def browse_directory(self, line_edit):
        """Browse for directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục", "./gallery")
        if dir_path:
            line_edit.setText(dir_path)
    
    def refresh_current_page(self):
        """Refresh current active page"""
        current_idx = self.stacked_widget.currentIndex()
        if current_idx == 0:  # Dashboard
            self.refresh_dashboard()
        elif current_idx == 1:  # Events
            self.refresh_events()
        elif current_idx == 2:  # Crops
            self.refresh_crops()
    
    def logout(self):
        """Logout - emit signal để main.py xử lý"""
        self.refresh_timer.stop()
        self.logout_signal.emit()
    
    def stop_video(self):
        """Stop video - Method để tương thích với main.py"""
        if hasattr(self, 'refresh_timer'):
            self.refresh_timer.stop()
    
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
        """Handle close"""
        self.refresh_timer.stop()
        event.accept()


# Backward compatibility
AdminPanel = ModernAdminPanel