#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Management Dialog - Quản lý người dùng (Admin only)
"""

import sys
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QFormLayout, QLineEdit,
    QComboBox, QDialogButtonBox, QAbstractItemView, QLabel
)
from PyQt5.QtCore import Qt

from database import Database


class UserManagementDialog(QDialog):
    """Dialog quản lý người dùng"""

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Quản lý Người dùng")
        self.setMinimumSize(600, 400)
        self.init_ui()
        self.load_users()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Username", "Role", "Created At", "Last Login"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # Set column widths
        self.table.setColumnWidth(0, 50)  # ID
        self.table.setColumnWidth(1, 150)  # Username
        self.table.setColumnWidth(2, 80)   # Role
        self.table.setColumnWidth(3, 150)  # Created At
        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()

        self.add_btn = QPushButton("➕ Thêm")
        self.add_btn.clicked.connect(self.add_user)
        button_layout.addWidget(self.add_btn)

        self.edit_btn = QPushButton("✏️ Sửa")
        self.edit_btn.clicked.connect(self.edit_user)
        button_layout.addWidget(self.edit_btn)

        self.delete_btn = QPushButton("🗑️ Xóa")
        self.delete_btn.clicked.connect(self.delete_user)
        button_layout.addWidget(self.delete_btn)

        button_layout.addStretch()

        self.refresh_btn = QPushButton("🔄 Làm mới")
        self.refresh_btn.clicked.connect(self.load_users)
        button_layout.addWidget(self.refresh_btn)

        self.close_btn = QPushButton("Đóng")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_users(self):
        """Load users from database"""
        try:
            users = self.db.get_all_users()
            self.table.setRowCount(len(users))

            for row, user in enumerate(users):
                self.table.setItem(row, 0, QTableWidgetItem(str(user['id'])))
                self.table.setItem(row, 1, QTableWidgetItem(user['username']))
                self.table.setItem(row, 2, QTableWidgetItem(user['role']))
                
                # Format created_at
                created_at = user.get('created_at')
                if created_at:
                    if isinstance(created_at, str):
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            created_at_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            created_at_str = str(created_at)
                    else:
                        created_at_str = str(created_at)
                else:
                    created_at_str = 'N/A'
                
                self.table.setItem(row, 3, QTableWidgetItem(created_at_str))
                
                # Format last_login
                last_login = user.get('last_login')
                if last_login:
                    if isinstance(last_login, str):
                        # Try to parse and format
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(last_login.replace('Z', '+00:00'))
                            last_login_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            last_login_str = str(last_login)
                    else:
                        last_login_str = str(last_login)
                else:
                    last_login_str = 'Chưa đăng nhập'
                
                self.table.setItem(row, 4, QTableWidgetItem(last_login_str))
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi tải danh sách người dùng: {str(e)}")

    def add_user(self):
        """Add new user"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Thêm người dùng mới")
        dialog.setMinimumSize(300, 200)

        layout = QFormLayout()

        username_input = QLineEdit()
        username_input.setPlaceholderText("Nhập username...")
        layout.addRow("Username:", username_input)

        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.Password)
        password_input.setPlaceholderText("Nhập password...")
        layout.addRow("Password:", password_input)

        role_combo = QComboBox()
        role_combo.addItems(["admin", "client"])
        layout.addRow("Role:", role_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            username = username_input.text().strip()
            password = password_input.text().strip()
            role = role_combo.currentText()

            # Validation
            if not username or not password:
                QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập đầy đủ thông tin!")
                return

            # Username validation
            if len(username) < 3:
                QMessageBox.warning(self, "Cảnh báo", "Username phải có ít nhất 3 ký tự!")
                return

            if not username.replace('_', '').replace('-', '').isalnum():
                QMessageBox.warning(self, "Cảnh báo", "Username chỉ được chứa chữ cái, số, dấu gạch dưới (_) và dấu gạch ngang (-)!")
                return

            # Password validation
            if len(password) < 4:
                QMessageBox.warning(self, "Cảnh báo", "Password phải có ít nhất 4 ký tự!")
                return

            if self.db.user_exists(username):
                QMessageBox.warning(self, "Cảnh báo", "Username đã tồn tại!")
                return

            try:
                user_id = self.db.add_user(username, password, role)
                QMessageBox.information(
                    self, 
                    "Thành công", 
                    f"Đã thêm người dùng mới!\n\n"
                    f"Username: {username}\n"
                    f"Role: {role}\n"
                    f"ID: {user_id}"
                )
                self.load_users()
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lỗi khi thêm người dùng: {str(e)}")

    def delete_user(self):
        """Delete selected user"""
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn người dùng để xóa!")
            return

        row = selected[0].row()
        user_id = int(self.table.item(row, 0).text())
        username = self.table.item(row, 1).text()
        user_role = self.table.item(row, 2).text()

        # Get current user from parent (if available)
        current_user_id = None
        if self.parent():
            try:
                # Try to get current user info from parent
                if hasattr(self.parent(), 'user_info'):
                    current_user_id = self.parent().user_info.get('id')
            except:
                pass

        # Prevent deleting yourself
        if current_user_id and user_id == current_user_id:
            QMessageBox.warning(self, "Cảnh báo", "Bạn không thể xóa chính mình!")
            return

        # Prevent deleting admin (already handled in database, but show warning)
        if user_role == 'admin':
            QMessageBox.warning(self, "Cảnh báo", "Không thể xóa tài khoản Admin!")
            return

        reply = QMessageBox.question(
            self,
            "Xác nhận",
            f"Bạn có chắc muốn xóa người dùng '{username}'?\n\n"
            f"⚠️ Hành động này không thể hoàn tác!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.db.delete_user(user_id)
                QMessageBox.information(self, "Thành công", f"Đã xóa người dùng '{username}'!")
                self.load_users()
            except ValueError as e:
                QMessageBox.warning(self, "Cảnh báo", str(e))
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lỗi khi xóa người dùng: {str(e)}")

    def edit_user(self):
        """Edit selected user"""
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn người dùng để sửa!")
            return

        row = selected[0].row()
        user_id = int(self.table.item(row, 0).text())
        current_username = self.table.item(row, 1).text()
        current_role = self.table.item(row, 2).text()

        # Get user info from database
        user = self.db.get_user(user_id)
        if not user:
            QMessageBox.warning(self, "Cảnh báo", "Không tìm thấy người dùng!")
            return

        # Create edit dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Sửa thông tin người dùng")
        dialog.setMinimumSize(400, 300)

        layout = QVBoxLayout()

        # Form layout
        form_layout = QFormLayout()

        # Username
        username_input = QLineEdit()
        username_input.setText(current_username)
        username_input.setPlaceholderText("Nhập username...")
        form_layout.addRow("Username:", username_input)

        # Password (optional - only change if provided)
        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.Password)
        password_input.setPlaceholderText("Để trống nếu không đổi password...")
        form_layout.addRow("Password mới:", password_input)

        # Role
        role_combo = QComboBox()
        role_combo.addItems(["admin", "client"])
        role_combo.setCurrentText(current_role)
        form_layout.addRow("Role:", role_combo)

        # Info label
        info_label = QLabel("💡 Để trống password nếu không muốn đổi mật khẩu")
        info_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        form_layout.addRow("", info_label)

        layout.addLayout(form_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            new_username = username_input.text().strip()
            new_password = password_input.text().strip()
            new_role = role_combo.currentText()

            # Validation
            if not new_username:
                QMessageBox.warning(self, "Cảnh báo", "Username không được để trống!")
                return

            if len(new_username) < 3:
                QMessageBox.warning(self, "Cảnh báo", "Username phải có ít nhất 3 ký tự!")
                return

            if not new_username.replace('_', '').replace('-', '').isalnum():
                QMessageBox.warning(self, "Cảnh báo", "Username chỉ được chứa chữ cái, số, dấu gạch dưới (_) và dấu gạch ngang (-)!")
                return

            # Check if username already exists (and not the current user)
            if new_username != current_username and self.db.user_exists(new_username):
                QMessageBox.warning(self, "Cảnh báo", "Username đã tồn tại!")
                return

            # Password validation (if provided)
            if new_password and len(new_password) < 4:
                QMessageBox.warning(self, "Cảnh báo", "Password phải có ít nhất 4 ký tự!")
                return

            try:
                # Update username if changed
                if new_username != current_username:
                    self.db.update_user(user_id, username=new_username)

                # Update role if changed
                if new_role != current_role:
                    self.db.update_user(user_id, role=new_role)

                # Update password if provided
                if new_password:
                    self.db.change_password(user_id, new_password)

                QMessageBox.information(self, "Thành công", "Đã cập nhật thông tin người dùng!")
                self.load_users()
            except ValueError as e:
                QMessageBox.warning(self, "Cảnh báo", str(e))
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lỗi khi cập nhật người dùng: {str(e)}")

