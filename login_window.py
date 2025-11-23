#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Login Window - Giao diện đăng nhập hiện đại
Split layout: Product info trên background, Login form trong card
"""

import sys
import logging as log
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QCheckBox, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QTimer
from PyQt5.QtGui import QFont, QColor, QLinearGradient, QPainter, QPaintEvent

from database import Database


class AnimatedBackgroundWidget(QWidget):
    """Animated gradient background"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation_offset = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_background)
        self.timer.start(50)
        self.setAutoFillBackground(True)
    
    def update_background(self):
        self.animation_offset = (self.animation_offset + 1) % 360
        self.update()
    
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Animated gradient
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        offset = self.animation_offset / 360.0
        
        color1 = QColor(79, 70, 229)
        color2 = QColor(99, 102, 241)
        color3 = QColor(139, 92, 246)
        
        gradient.setColorAt(0, color1)
        gradient.setColorAt(0.5 + offset * 0.2, color2)
        gradient.setColorAt(1, color3)
        
        painter.fillRect(self.rect(), gradient)
        
        # Floating circles
        painter.setOpacity(0.05)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        
        for i in range(0, self.width() + 200, 200):
            x = (i + self.animation_offset * 2) % (self.width() + 200)
            painter.drawEllipse(int(x - 100), int(self.height() * 0.2), 200, 200)
        
        super().paintEvent(event)


class StyledLineEdit(QLineEdit):
    """Modern styled line edit with icon"""
    
    def __init__(self, icon="", placeholder="", parent=None):
        super().__init__(parent)
        self.icon = icon
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(52)
        self.setStyleSheet("""
            QLineEdit {
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                padding: 14px 18px 14px 50px;
                font-size: 15px;
                background-color: #f8fafc;
                color: #0f172a;
                selection-background-color: #6366f1;
            }
            QLineEdit:focus {
                border: 2px solid #6366f1;
                background-color: white;
                outline: none;
            }
            QLineEdit:hover {
                border: 2px solid #cbd5e0;
                background-color: white;
            }
        """)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.icon:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Icon background
            painter.setBrush(QColor(99, 102, 241, 30))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(12, (self.height() - 28) // 2, 28, 28)
            
            # Icon text
            font = QFont()
            font.setPointSize(15)
            painter.setFont(font)
            painter.setPen(QColor(99, 102, 241))
            painter.drawText(12, (self.height() - 14) // 2 + 14, 28, 28, Qt.AlignCenter, self.icon)


class ModernLoginWindow(QWidget):
    """Modern Login Window - Split Layout"""

    login_successful = pyqtSignal(dict)

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Face Recognition System - Login")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        
        self.init_ui()
        self.setup_animations()

    def init_ui(self):
        """Initialize split layout UI"""
        # Main container
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Background
        self.background = AnimatedBackgroundWidget()
        bg_layout = QHBoxLayout()
        bg_layout.setContentsMargins(80, 60, 80, 60)
        bg_layout.setSpacing(60)
        
        # ===== LEFT SIDE: Product Information =====
        left_section = QVBoxLayout()
        left_section.setSpacing(40)
        
        # Logo & Branding
        branding = QVBoxLayout()
        branding.setSpacing(20)
        
        # Large Logo
        logo = QLabel("👁️")
        logo_font = QFont()
        logo_font.setPointSize(64)
        logo.setFont(logo_font)
        logo.setStyleSheet("color: white; background: transparent;")
        logo.setAlignment(Qt.AlignLeft)
        branding.addWidget(logo)
        
        # Title
        title = QLabel("Face Recognition System")
        title_font = QFont()
        title_font.setPointSize(38)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("""
            color: white;
            background: transparent;
            letter-spacing: 1px;
        """)
        title.setWordWrap(True)
        branding.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Hệ thống Nhận diện Khách VIP Thông minh")
        subtitle_font = QFont()
        subtitle_font.setPointSize(18)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.9); background: transparent;")
        subtitle.setWordWrap(True)
        branding.addWidget(subtitle)
        
        left_section.addLayout(branding)
        left_section.addSpacing(20)
        
        # Features
        features_title = QLabel("Tính năng nổi bật")
        features_title_font = QFont()
        features_title_font.setPointSize(20)
        features_title_font.setBold(True)
        features_title.setFont(features_title_font)
        features_title.setStyleSheet("color: white; background: transparent;")
        left_section.addWidget(features_title)
        
        features_list = [
            ("🎯", "Nhận diện chính xác", "Công nghệ AI tiên tiến với độ chính xác cao"),
            ("⚡", "Xử lý nhanh chóng", "Phản hồi tức thì trong vòng 1 giây"),
            ("🔒", "Bảo mật tuyệt đối", "Mã hóa dữ liệu end-to-end"),
            ("📊", "Thống kê chi tiết", "Báo cáo và phân tích dữ liệu khách hàng"),
            ("👥", "Quản lý đa cấp", "Phân quyền Admin và Client linh hoạt"),
        ]
        
        features_container = QVBoxLayout()
        features_container.setSpacing(18)
        
        for icon, title_text, desc in features_list:
            feature_item = QHBoxLayout()
            feature_item.setSpacing(15)
            
            # Icon
            icon_label = QLabel(icon)
            icon_label.setStyleSheet("color: white; font-size: 28px; background: transparent;")
            icon_label.setFixedWidth(40)
            feature_item.addWidget(icon_label)
            
            # Text
            text_container = QVBoxLayout()
            text_container.setSpacing(4)
            
            feature_title = QLabel(title_text)
            feature_title.setStyleSheet("color: white; font-size: 16px; font-weight: 600; background: transparent;")
            text_container.addWidget(feature_title)
            
            feature_desc = QLabel(desc)
            feature_desc.setStyleSheet("color: rgba(255, 255, 255, 0.8); font-size: 13px; background: transparent;")
            feature_desc.setWordWrap(True)
            text_container.addWidget(feature_desc)
            
            feature_item.addLayout(text_container)
            features_container.addLayout(feature_item)
        
        left_section.addLayout(features_container)
        left_section.addStretch()
        
        # Footer info
        footer_info = QLabel("© 2024 Face Recognition System | Version 2.0")
        footer_info.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 12px; background: transparent;")
        left_section.addWidget(footer_info)
        
        # ===== RIGHT SIDE: Login Card =====
        right_section = QVBoxLayout()
        right_section.setAlignment(Qt.AlignCenter)
        
        # Login card
        self.login_card = QFrame()
        self.login_card.setFixedSize(420, 560)
        self.login_card.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.98);
                border-radius: 24px;
            }
        """)
        
        # Card layout
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(0)
        
        # Card header
        header = QVBoxLayout()
        header.setSpacing(12)
        header.setAlignment(Qt.AlignCenter)
        
        welcome = QLabel("Đăng nhập")
        welcome_font = QFont()
        welcome_font.setPointSize(26)
        welcome_font.setBold(True)
        welcome.setFont(welcome_font)
        welcome.setAlignment(Qt.AlignCenter)
        welcome.setStyleSheet("color: #0f172a; background: transparent;")
        header.addWidget(welcome)
        
        desc = QLabel("Vui lòng nhập thông tin để tiếp tục")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #64748b; font-size: 14px; background: transparent;")
        header.addWidget(desc)
        
        card_layout.addLayout(header)
        card_layout.addSpacing(36)
        
        # Form
        form = QVBoxLayout()
        form.setSpacing(20)
        
        # Username
        username_label = QLabel("Tên đăng nhập")
        username_label.setStyleSheet("color: #334155; font-weight: 600; font-size: 13px;")
        form.addWidget(username_label)
        
        self.username_input = StyledLineEdit("👤", "Nhập tên đăng nhập")
        self.username_input.textChanged.connect(self.clear_errors)
        form.addWidget(self.username_input)
        
        self.username_error = QLabel("")
        self.username_error.setStyleSheet("color: #ef4444; font-size: 11px; margin-top: 4px;")
        self.username_error.setVisible(False)
        form.addWidget(self.username_error)
        
        form.addSpacing(8)
        
        # Password
        password_label = QLabel("Mật khẩu")
        password_label.setStyleSheet("color: #334155; font-weight: 600; font-size: 13px;")
        form.addWidget(password_label)
        
        self.password_input = StyledLineEdit("🔒", "Nhập mật khẩu")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.textChanged.connect(self.clear_errors)
        self.password_input.returnPressed.connect(self.login)
        form.addWidget(self.password_input)
        
        self.password_error = QLabel("")
        self.password_error.setStyleSheet("color: #ef4444; font-size: 11px; margin-top: 4px;")
        self.password_error.setVisible(False)
        form.addWidget(self.password_error)
        
        form.addSpacing(8)
        
        # Remember me
        self.remember_checkbox = QCheckBox("Ghi nhớ đăng nhập")
        self.remember_checkbox.setStyleSheet("""
            QCheckBox {
                color: #64748b;
                font-size: 13px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #cbd5e0;
                border-radius: 5px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #6366f1;
                border-color: #6366f1;
            }
            QCheckBox::indicator:hover {
                border-color: #6366f1;
            }
        """)
        form.addWidget(self.remember_checkbox)
        
        form.addSpacing(16)
        
        # Login button
        self.login_btn = QPushButton("Đăng nhập")
        self.login_btn.setMinimumHeight(52)
        self.login_btn.setCursor(Qt.PointingHandCursor)
        self.login_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #8b5cf6);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4f46e5, stop:1 #7c3aed);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4338ca, stop:1 #6d28d9);
            }
            QPushButton:disabled {
                background: #e2e8f0;
                color: #cbd5e1;
            }
        """)
        self.login_btn.clicked.connect(self.login)
        form.addWidget(self.login_btn)
        
        # Loading bar
        self.loading_bar = QProgressBar()
        self.loading_bar.setMinimumHeight(3)
        self.loading_bar.setMaximum(0)
        self.loading_bar.setVisible(False)
        self.loading_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: rgba(226, 232, 240, 0.5);
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #6366f1;
                border-radius: 2px;
            }
        """)
        form.addWidget(self.loading_bar)
        
        form.addSpacing(24)
        
        # Demo accounts info
        demo_info = QFrame()
        demo_info.setStyleSheet("""
            QFrame {
                background-color: #f1f5f9;
                border-radius: 10px;
                border: 1px solid #e2e8f0;
            }
        """)
        demo_layout = QVBoxLayout()
        demo_layout.setContentsMargins(16, 14, 16, 14)
        demo_layout.setSpacing(8)
        
        demo_title = QLabel("💡 Tài khoản demo")
        demo_title.setStyleSheet("color: #475569; font-weight: 600; font-size: 12px;")
        demo_layout.addWidget(demo_title)
        
        demo_content = QLabel(
            "<div style='line-height: 1.6;'>"
            "<span style='color: #6366f1;'>Admin:</span> <span style='color: #64748b;'>admin / 1234</span><br>"
            "<span style='color: #6366f1;'>Client:</span> <span style='color: #64748b;'>client / 1234</span>"
            "</div>"
        )
        demo_content.setStyleSheet("font-size: 12px;")
        demo_layout.addWidget(demo_content)
        
        demo_info.setLayout(demo_layout)
        form.addWidget(demo_info)
        
        card_layout.addLayout(form)
        card_layout.addStretch()
        
        self.login_card.setLayout(card_layout)
        right_section.addWidget(self.login_card)
        
        # Combine layouts
        bg_layout.addLayout(left_section, 5)  # 50% width
        bg_layout.addLayout(right_section, 4)  # 40% width
        
        self.background.setLayout(bg_layout)
        main_layout.addWidget(self.background)
        
        self.setLayout(main_layout)
        self.username_input.setFocus()

    def setup_animations(self):
        """Setup entrance animations"""
        self.setWindowOpacity(0.0)
        
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(400)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        QTimer.singleShot(100, self.fade_anim.start)

    def clear_errors(self):
        """Clear error messages"""
        self.username_error.setVisible(False)
        self.password_error.setVisible(False)

    def login(self):
        """Handle login"""
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        
        self.clear_errors()
        
        has_error = False
        if not username:
            self.username_error.setText("⚠️ Vui lòng nhập tên đăng nhập")
            self.username_error.setVisible(True)
            has_error = True
        
        if not password:
            self.password_error.setText("⚠️ Vui lòng nhập mật khẩu")
            self.password_error.setVisible(True)
            has_error = True
        
        if has_error:
            self.shake_card()
            return

        self.login_btn.setEnabled(False)
        self.login_btn.setText("Đang xác thực...")
        self.loading_bar.setVisible(True)
        QTimer.singleShot(100, lambda: self._perform_login(username, password))

    def _perform_login(self, username: str, password: str):
        """Perform actual login"""
        try:
            user_info = self.db.authenticate_user(username, password)
            if user_info:
                log.info(f"User logged in: {username} (Role: {user_info['role']})")
                self.success_animation()
                QTimer.singleShot(400, lambda: self.login_successful.emit(user_info))
            else:
                self.username_error.setText("❌ Tên đăng nhập hoặc mật khẩu không đúng")
                self.username_error.setVisible(True)
                self.password_input.clear()
                self.shake_card()
        except Exception as e:
            log.error(f"Login error: {e}")
            self.username_error.setText(f"❌ Lỗi: {str(e)[:40]}")
            self.username_error.setVisible(True)
            self.shake_card()
        finally:
            self.login_btn.setEnabled(True)
            self.login_btn.setText("Đăng nhập")
            self.loading_bar.setVisible(False)

    def shake_card(self):
        """Shake animation for error"""
        anim = QPropertyAnimation(self.login_card, b"geometry")
        anim.setDuration(400)
        anim.setEasingCurve(QEasingCurve.InOutQuad)
        
        geo = self.login_card.geometry()
        anim.setKeyValueAt(0, geo)
        anim.setKeyValueAt(0.2, QRect(geo.x() - 12, geo.y(), geo.width(), geo.height()))
        anim.setKeyValueAt(0.4, QRect(geo.x() + 12, geo.y(), geo.width(), geo.height()))
        anim.setKeyValueAt(0.6, QRect(geo.x() - 8, geo.y(), geo.width(), geo.height()))
        anim.setKeyValueAt(0.8, QRect(geo.x() + 8, geo.y(), geo.width(), geo.height()))
        anim.setKeyValueAt(1, geo)
        anim.start()

    def success_animation(self):
        """Success animation"""
        self.login_btn.setText("✓ Thành công!")
        self.login_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #059669);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: bold;
            }
        """)

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
        """Handle ESC key"""
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)
    def clear_fields(self):
        """Clear username and password fields"""
        if hasattr(self, 'username_input'):
            self.username_input.clear()
        if hasattr(self, 'password_input'):
            self.password_input.clear()
        if hasattr(self, 'username_input'):
            self.username_input.setFocus()

# Backward compatibility
LoginWindow = ModernLoginWindow