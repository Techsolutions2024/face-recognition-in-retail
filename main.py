#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point - Face Recognition System with User Authentication
Điều phối Login, Admin Panel, và Client Panel
"""

import sys
import logging as log
from PyQt5.QtWidgets import QApplication

from database import Database
from login_window import LoginWindow
from admin_panel import AdminPanel
from client_panel import ClientPanel

# Cấu hình logging
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)


class MainApplication:
    """Main application để quản lý các window"""

    def __init__(self):
        # Initialize database
        self.db = Database("facere.db")

        # Create login window
        self.login_window = LoginWindow(self.db)
        self.login_window.login_successful.connect(self.on_login_successful)

        # Admin and Client panels (created on demand)
        self.admin_panel = None
        self.client_panel = None

        # Show login window
        self.login_window.show()

    def on_login_successful(self, user_info):
        """Handle successful login"""
        role = user_info['role']

        # Hide login window
        self.login_window.hide()

        if role == 'admin':
            # Create admin panel if not exists
            if self.admin_panel is None:
                self.admin_panel = AdminPanel(user_info, self.db)
                self.admin_panel.logout_signal.connect(self.logout)

            # Show admin panel
            self.admin_panel.show()

        elif role == 'client':
            # Create client panel if not exists
            if self.client_panel is None:
                self.client_panel = ClientPanel(user_info, self.db)
                self.client_panel.logout_signal.connect(self.logout)

            # Show client panel
            self.client_panel.show()

    def logout(self):
        """Logout and return to login"""
        # Hide panels
        if self.admin_panel:
            self.admin_panel.hide()
            self.admin_panel.stop_video()
        if self.client_panel:
            self.client_panel.hide()
            self.client_panel.stop_video()

        # Clear login fields and show login window
        self.login_window.clear_fields()
        self.login_window.show()


def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Create main application
    main_app = MainApplication()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

