#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Layer for Face Recognition System
SQLite database với CRUD operations
"""

import sqlite3
import logging as log
from pathlib import Path
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from models import Camera, Customer, Event, Crop, Visit


class Database:
    """SQLite database manager"""

    def __init__(self, db_path: str = "facere.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Get database connection as context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self):
        """Initialize database tables"""
        log.info(f"Initializing database: {self.db_path}")

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create cameras table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_type TEXT DEFAULT 'webcam',
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create customers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    segment TEXT DEFAULT 'regular',
                    total_visits INTEGER DEFAULT 0,
                    last_visit_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')

            # Create events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    customer_id INTEGER,
                    customer_name TEXT,
                    camera_id INTEGER,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(id),
                    FOREIGN KEY (camera_id) REFERENCES cameras(id)
                )
            ''')

            # Create crops table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER,
                    event_id INTEGER,
                    file_path TEXT NOT NULL,
                    bbox TEXT,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers(id),
                    FOREIGN KEY (event_id) REFERENCES events(id)
                )
            ''')

            # Create visits table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    dwell_time_seconds INTEGER DEFAULT 0,
                    camera_id INTEGER,
                    FOREIGN KEY (customer_id) REFERENCES customers(id),
                    FOREIGN KEY (camera_id) REFERENCES cameras(id)
                )
            ''')

            # Create users table for authentication
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'client',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')

            # Create settings table for model configuration
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_customer ON events(customer_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_crops_customer ON crops(customer_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_visits_customer ON visits(customer_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_customers_face_id ON customers(face_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')

            # Initialize default users if not exist
            cursor.execute('SELECT COUNT(*) FROM users')
            user_count = cursor.fetchone()[0]
            if user_count == 0:
                # Create default admin user
                cursor.execute(
                    'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                    ('admin', '1234', 'admin')
                )
                # Create default client user
                cursor.execute(
                    'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                    ('client', '1234', 'client')
                )
                log.info("Default users created: admin/1234, client/1234")

            log.info("Database initialized successfully")

    # ==================== CAMERA OPERATIONS ====================

    def add_camera(self, name: str, source: str, source_type: str = "webcam") -> int:
        """Add new camera"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO cameras (name, source, source_type) VALUES (?, ?, ?)',
                (name, source, source_type)
            )
            return cursor.lastrowid

    def get_camera(self, camera_id: int) -> Optional[Camera]:
        """Get camera by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM cameras WHERE id = ?', (camera_id,))
            row = cursor.fetchone()
            if row:
                return Camera.from_dict(dict(row))
            return None

    def get_all_cameras(self) -> List[Camera]:
        """Get all cameras"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM cameras ORDER BY id')
            rows = cursor.fetchall()
            return [Camera.from_dict(dict(row)) for row in rows]

    def update_camera(self, camera_id: int, **kwargs):
        """Update camera fields"""
        if not kwargs:
            return

        fields = ', '.join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [camera_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'UPDATE cameras SET {fields} WHERE id = ?', values)

    def delete_camera(self, camera_id: int):
        """Delete camera"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))

    # ==================== CUSTOMER OPERATIONS ====================

    def add_customer(self, face_id: str, name: str, segment: str = "regular") -> int:
        """Add new customer"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO customers (face_id, name, segment) VALUES (?, ?, ?)',
                (face_id, name, segment)
            )
            return cursor.lastrowid

    def get_customer(self, customer_id: int) -> Optional[Customer]:
        """Get customer by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM customers WHERE id = ?', (customer_id,))
            row = cursor.fetchone()
            if row:
                return Customer.from_dict(dict(row))
            return None

    def get_customer_by_face_id(self, face_id: str) -> Optional[Customer]:
        """Get customer by face_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM customers WHERE face_id = ?', (face_id,))
            row = cursor.fetchone()
            if row:
                return Customer.from_dict(dict(row))
            return None

    def get_customer_by_name(self, name: str) -> Optional[Customer]:
        """Get customer by name"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM customers WHERE name = ?', (name,))
            row = cursor.fetchone()
            if row:
                return Customer.from_dict(dict(row))
            return None

    def get_all_customers(self, segment: Optional[str] = None) -> List[Customer]:
        """Get all customers, optionally filtered by segment"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if segment:
                cursor.execute('SELECT * FROM customers WHERE segment = ? ORDER BY name', (segment,))
            else:
                cursor.execute('SELECT * FROM customers ORDER BY name')
            rows = cursor.fetchall()
            return [Customer.from_dict(dict(row)) for row in rows]

    def update_customer(self, customer_id: int, **kwargs):
        """Update customer fields"""
        if not kwargs:
            return

        fields = ', '.join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [customer_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'UPDATE customers SET {fields} WHERE id = ?', values)

    def increment_customer_visits(self, customer_id: int):
        """Increment customer visit count"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE customers SET total_visits = total_visits + 1, last_visit_date = ? WHERE id = ?',
                (datetime.now(), customer_id)
            )

    def delete_customer(self, customer_id: int):
        """Delete customer and all related records (cascade delete)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Delete related records first (SQLite doesn't support CASCADE DELETE by default)
            # 1. Delete crops
            cursor.execute('DELETE FROM crops WHERE customer_id = ?', (customer_id,))
            # 2. Delete events
            cursor.execute('DELETE FROM events WHERE customer_id = ?', (customer_id,))
            # 3. Delete visits
            cursor.execute('DELETE FROM visits WHERE customer_id = ?', (customer_id,))
            # 4. Finally delete customer
            cursor.execute('DELETE FROM customers WHERE id = ?', (customer_id,))
            conn.commit()

    # ==================== EVENT OPERATIONS ====================

    def add_event(self, event_type: str, customer_name: str, customer_id: Optional[int] = None,
                  camera_id: Optional[int] = None, confidence: float = 0.0,
                  metadata: Optional[str] = None) -> int:
        """Add new event"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO events
                   (event_type, customer_id, customer_name, camera_id, confidence, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (event_type, customer_id, customer_name, camera_id, confidence, metadata)
            )
            return cursor.lastrowid

    def get_event(self, event_id: int) -> Optional[Event]:
        """Get event by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM events WHERE id = ?', (event_id,))
            row = cursor.fetchone()
            if row:
                return Event.from_dict(dict(row))
            return None

    def update_event_metadata(self, event_id: int, metadata: Dict[str, Any]) -> bool:
        """Update event metadata"""
        import json
        metadata_str = json.dumps(metadata) if metadata else None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE events SET metadata = ? WHERE id = ?',
                (metadata_str, event_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_recent_events(self, limit: int = 20, event_type: Optional[str] = None) -> List[Event]:
        """Get recent events"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if event_type:
                cursor.execute(
                    'SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?',
                    (event_type, limit)
                )
            else:
                cursor.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (limit,))
            rows = cursor.fetchall()
            return [Event.from_dict(dict(row)) for row in rows]

    def get_events_by_customer(self, customer_id: int, limit: int = 50) -> List[Event]:
        """Get events for a customer"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM events WHERE customer_id = ? ORDER BY timestamp DESC LIMIT ?',
                (customer_id, limit)
            )
            rows = cursor.fetchall()
            return [Event.from_dict(dict(row)) for row in rows]

    def get_event_count_today(self) -> int:
        """Get event count for today"""
        today = date.today()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT COUNT(*) FROM events
                   WHERE DATE(timestamp) = ? AND event_type IN ('recognized', 'unknown')''',
                (today,)
            )
            return cursor.fetchone()[0]

    def get_event_stats(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> Dict[str, int]:
        """Get event statistics"""
        if not start_date:
            start_date = date.today()
        if not end_date:
            end_date = date.today()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT event_type, COUNT(*) as count
                   FROM events
                   WHERE DATE(timestamp) BETWEEN ? AND ?
                   GROUP BY event_type''',
                (start_date, end_date)
            )
            return {row['event_type']: row['count'] for row in cursor.fetchall()}

    # ==================== CROP OPERATIONS ====================

    def add_crop(self, file_path: str, customer_id: Optional[int] = None,
                 event_id: Optional[int] = None, bbox: Optional[str] = None,
                 confidence: float = 0.0) -> int:
        """Add new crop"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO crops
                   (customer_id, event_id, file_path, bbox, confidence)
                   VALUES (?, ?, ?, ?, ?)''',
                (customer_id, event_id, file_path, bbox, confidence)
            )
            return cursor.lastrowid

    def get_crop(self, crop_id: int) -> Optional[Crop]:
        """Get crop by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM crops WHERE id = ?', (crop_id,))
            row = cursor.fetchone()
            if row:
                return Crop.from_dict(dict(row))
            return None

    def get_crops_by_customer(self, customer_id: int, limit: int = 10) -> List[Crop]:
        """Get crops for a customer"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM crops WHERE customer_id = ? ORDER BY timestamp DESC LIMIT ?',
                (customer_id, limit)
            )
            rows = cursor.fetchall()
            return [Crop.from_dict(dict(row)) for row in rows]

    def get_recent_crops(self, limit: int = 20) -> List[Crop]:
        """Get recent crops"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM crops ORDER BY timestamp DESC LIMIT ?', (limit,))
            rows = cursor.fetchall()
            return [Crop.from_dict(dict(row)) for row in rows]

    # ==================== VISIT OPERATIONS ====================

    def add_visit(self, customer_id: int, camera_id: int) -> int:
        """Add new visit (entry)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO visits
                   (customer_id, entry_time, camera_id)
                   VALUES (?, ?, ?)''',
                (customer_id, datetime.now(), camera_id)
            )
            return cursor.lastrowid

    def get_visit(self, visit_id: int) -> Optional[Visit]:
        """Get visit by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM visits WHERE id = ?', (visit_id,))
            row = cursor.fetchone()
            if row:
                return Visit.from_dict(dict(row))
            return None

    def get_active_visit(self, customer_id: int) -> Optional[Visit]:
        """Get active visit for customer (not exited)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM visits WHERE customer_id = ? AND exit_time IS NULL ORDER BY entry_time DESC LIMIT 1',
                (customer_id,)
            )
            row = cursor.fetchone()
            if row:
                return Visit.from_dict(dict(row))
            return None

    def end_visit(self, visit_id: int):
        """End visit (set exit time)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Get entry time
            cursor.execute('SELECT entry_time FROM visits WHERE id = ?', (visit_id,))
            row = cursor.fetchone()
            if row:
                entry_time = datetime.fromisoformat(row['entry_time'])
                exit_time = datetime.now()
                dwell_time = int((exit_time - entry_time).total_seconds())

                cursor.execute(
                    'UPDATE visits SET exit_time = ?, dwell_time_seconds = ? WHERE id = ?',
                    (exit_time, dwell_time, visit_id)
                )

    def get_visits_by_customer(self, customer_id: int, limit: int = 50) -> List[Visit]:
        """Get visits for a customer"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM visits WHERE customer_id = ? ORDER BY entry_time DESC LIMIT ?',
                (customer_id, limit)
            )
            rows = cursor.fetchall()
            return [Visit.from_dict(dict(row)) for row in rows]

    # ==================== USER OPERATIONS ====================

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user info"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, username, role FROM users WHERE username = ? AND password = ?',
                (username, password)
            )
            row = cursor.fetchone()
            if row:
                # Update last login
                cursor.execute(
                    'UPDATE users SET last_login = ? WHERE id = ?',
                    (datetime.now(), row['id'])
                )
                return {
                    'id': row['id'],
                    'username': row['username'],
                    'role': row['role']
                }
            return None

    def add_user(self, username: str, password: str, role: str = "client") -> int:
        """Add new user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                (username, password, role)
            )
            return cursor.lastrowid

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, username, role, created_at, last_login FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, username, role, created_at, last_login FROM users ORDER BY username')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_user(self, user_id: int, **kwargs):
        """Update user fields"""
        if not kwargs:
            return

        # Don't allow updating password without explicit method
        if 'password' in kwargs:
            raise ValueError("Use change_password() method to update password")

        fields = ', '.join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [user_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'UPDATE users SET {fields} WHERE id = ?', values)

    def change_password(self, user_id: int, new_password: str):
        """Change user password"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET password = ? WHERE id = ?', (new_password, user_id))

    def delete_user(self, user_id: int):
        """Delete user (prevent deleting admin)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Check if user is admin
            cursor.execute('SELECT role FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            if row and row['role'] == 'admin':
                raise ValueError("Cannot delete admin user")
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))

    def user_exists(self, username: str) -> bool:
        """Check if username exists"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
            return cursor.fetchone()[0] > 0

    # ==================== SETTINGS OPERATIONS ====================

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get setting value by key"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                return row['value']
            return default

    def set_setting(self, key: str, value: str):
        """Set setting value"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))

    def get_model_config(self) -> Dict[str, str]:
        """Get model configuration"""
        return {
            'model_fd_path': self.get_setting('model_fd_path', './models/face-detection-retail-0004.xml'),
            'model_lm_path': self.get_setting('model_lm_path', './models/landmarks-regression-retail-0009.xml'),
            'model_reid_path': self.get_setting('model_reid_path', './models/face-reidentification-retail-0095.xml'),
            'gallery_path': self.get_setting('gallery_path', './gallery')
        }

    def save_model_config(self, model_fd_path: str, model_lm_path: str, 
                         model_reid_path: str, gallery_path: str):
        """Save model configuration"""
        self.set_setting('model_fd_path', model_fd_path)
        self.set_setting('model_lm_path', model_lm_path)
        self.set_setting('model_reid_path', model_reid_path)
        self.set_setting('gallery_path', gallery_path)
        log.info(f"Model configuration saved: FD={model_fd_path}, LM={model_lm_path}, ReID={model_reid_path}, Gallery={gallery_path}")

    def get_detection_cooldown(self) -> float:
        """Get detection cooldown time in seconds"""
        cooldown_str = self.get_setting('detection_cooldown', '5.0')
        try:
            return float(cooldown_str)
        except ValueError:
            return 5.0  # Default 5 seconds

    def set_detection_cooldown(self, cooldown_seconds: float):
        """Set detection cooldown time in seconds"""
        self.set_setting('detection_cooldown', str(cooldown_seconds))
        log.info(f"Detection cooldown set to {cooldown_seconds} seconds")

    def get_revisit_threshold(self) -> float:
        """Get revisit threshold time in hours"""
        threshold_str = self.get_setting('revisit_threshold', '3.0')
        try:
            return float(threshold_str)
        except ValueError:
            return 3.0  # Default 3 hours

    def set_revisit_threshold(self, threshold_hours: float):
        """Set revisit threshold time in hours"""
        self.set_setting('revisit_threshold', str(threshold_hours))
        log.info(f"Revisit threshold set to {threshold_hours} hours")

    def get_capture_interval(self) -> float:
        """Get capture interval time in seconds (thời gian giữa các lần chụp)"""
        interval_str = self.get_setting('capture_interval', '2.0')
        try:
            return float(interval_str)
        except ValueError:
            return 2.0  # Default 2 seconds

    def set_capture_interval(self, interval_seconds: float):
        """Set capture interval time in seconds"""
        self.set_setting('capture_interval', str(interval_seconds))
        log.info(f"Capture interval set to {interval_seconds} seconds")

    # ==================== UTILITY METHODS ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total customers
            cursor.execute('SELECT COUNT(*) FROM customers')
            total_customers = cursor.fetchone()[0]

            # Total events today
            today = date.today()
            cursor.execute(
                'SELECT COUNT(*) FROM events WHERE DATE(timestamp) = ?',
                (today,)
            )
            events_today = cursor.fetchone()[0]

            # VIP customers
            cursor.execute('SELECT COUNT(*) FROM customers WHERE segment = "vip"')
            vip_count = cursor.fetchone()[0]

            # Active cameras
            cursor.execute('SELECT COUNT(*) FROM cameras WHERE status = "active"')
            active_cameras = cursor.fetchone()[0]

            return {
                'total_customers': total_customers,
                'events_today': events_today,
                'vip_count': vip_count,
                'active_cameras': active_cameras
            }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            log.info("Database connection closed")
