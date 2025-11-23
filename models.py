#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models for Face Recognition System
Định nghĩa các data structures cho database
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class Camera:
    """Camera model"""
    id: Optional[int] = None
    name: str = ""
    source: str = "0"  # '0', '1', 'video.mp4', 'rtsp://...'
    source_type: str = "webcam"  # 'webcam', 'video', 'rtsp', 'ip'
    status: str = "active"  # 'active', 'offline', 'degraded'
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'source': self.source,
            'source_type': self.source_type,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Camera':
        """Create from dictionary"""
        created_at = None
        if data.get('created_at'):
            if isinstance(data['created_at'], str):
                created_at = datetime.fromisoformat(data['created_at'])
            else:
                created_at = data['created_at']

        return cls(
            id=data.get('id'),
            name=data.get('name', ''),
            source=data.get('source', '0'),
            source_type=data.get('source_type', 'webcam'),
            status=data.get('status', 'active'),
            created_at=created_at
        )


@dataclass
class Customer:
    """Customer model"""
    id: Optional[int] = None
    face_id: str = ""  # e.g., 'John-0'
    name: str = ""
    segment: str = "regular"  # 'regular', 'vip', 'new', 'blacklist'
    total_visits: int = 0
    last_visit_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'face_id': self.face_id,
            'name': self.name,
            'segment': self.segment,
            'total_visits': self.total_visits,
            'last_visit_date': self.last_visit_date.isoformat() if self.last_visit_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': json.dumps(self.metadata) if self.metadata else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Customer':
        """Create from dictionary"""
        last_visit_date = None
        if data.get('last_visit_date'):
            if isinstance(data['last_visit_date'], str):
                last_visit_date = datetime.fromisoformat(data['last_visit_date'])
            else:
                last_visit_date = data['last_visit_date']

        created_at = None
        if data.get('created_at'):
            if isinstance(data['created_at'], str):
                created_at = datetime.fromisoformat(data['created_at'])
            else:
                created_at = data['created_at']

        metadata = None
        if data.get('metadata'):
            if isinstance(data['metadata'], str):
                metadata = json.loads(data['metadata'])
            else:
                metadata = data['metadata']

        return cls(
            id=data.get('id'),
            face_id=data.get('face_id', ''),
            name=data.get('name', ''),
            segment=data.get('segment', 'regular'),
            total_visits=data.get('total_visits', 0),
            last_visit_date=last_visit_date,
            created_at=created_at,
            metadata=metadata
        )

    def is_vip(self) -> bool:
        """Check if customer is VIP"""
        return self.segment == 'vip'

    def is_blacklist(self) -> bool:
        """Check if customer is blacklisted"""
        return self.segment == 'blacklist'


@dataclass
class Event:
    """Event model"""
    id: Optional[int] = None
    event_type: str = ""  # 'entry', 'exit', 'recognized', 'unknown', 'vip_detected', etc.
    customer_id: Optional[int] = None
    customer_name: str = ""
    camera_id: Optional[int] = None
    confidence: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'customer_id': self.customer_id,
            'customer_name': self.customer_name,
            'camera_id': self.camera_id,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': json.dumps(self.metadata) if self.metadata else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create from dictionary"""
        timestamp = None
        if data.get('timestamp'):
            if isinstance(data['timestamp'], str):
                timestamp = datetime.fromisoformat(data['timestamp'])
            else:
                timestamp = data['timestamp']

        metadata = None
        if data.get('metadata'):
            if isinstance(data['metadata'], str):
                metadata = json.loads(data['metadata'])
            else:
                metadata = data['metadata']

        return cls(
            id=data.get('id'),
            event_type=data.get('event_type', ''),
            customer_id=data.get('customer_id'),
            customer_name=data.get('customer_name', ''),
            camera_id=data.get('camera_id'),
            confidence=data.get('confidence', 0.0),
            timestamp=timestamp,
            metadata=metadata
        )

    def get_display_icon(self) -> str:
        """Get icon for event type"""
        icons = {
            'entry': '🚪',
            'exit': '👋',
            'recognized': '✓',
            'unknown': '❓',
            'vip_detected': '⭐',
            'blacklist': '⚠️',
            'new_customer': '🆕'
        }
        return icons.get(self.event_type, '📋')


@dataclass
class Crop:
    """Face crop model"""
    id: Optional[int] = None
    customer_id: Optional[int] = None
    event_id: Optional[int] = None
    file_path: str = ""
    bbox: Optional[Dict[str, int]] = None  # {'x': ..., 'y': ..., 'w': ..., 'h': ...}
    confidence: float = 0.0
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'event_id': self.event_id,
            'file_path': self.file_path,
            'bbox': json.dumps(self.bbox) if self.bbox else None,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Crop':
        """Create from dictionary"""
        timestamp = None
        if data.get('timestamp'):
            if isinstance(data['timestamp'], str):
                timestamp = datetime.fromisoformat(data['timestamp'])
            else:
                timestamp = data['timestamp']

        bbox = None
        if data.get('bbox'):
            if isinstance(data['bbox'], str):
                bbox = json.loads(data['bbox'])
            else:
                bbox = data['bbox']

        return cls(
            id=data.get('id'),
            customer_id=data.get('customer_id'),
            event_id=data.get('event_id'),
            file_path=data.get('file_path', ''),
            bbox=bbox,
            confidence=data.get('confidence', 0.0),
            timestamp=timestamp
        )


@dataclass
class Visit:
    """Visit tracking model"""
    id: Optional[int] = None
    customer_id: Optional[int] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    dwell_time_seconds: int = 0
    camera_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'dwell_time_seconds': self.dwell_time_seconds,
            'camera_id': self.camera_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Visit':
        """Create from dictionary"""
        entry_time = None
        if data.get('entry_time'):
            if isinstance(data['entry_time'], str):
                entry_time = datetime.fromisoformat(data['entry_time'])
            else:
                entry_time = data['entry_time']

        exit_time = None
        if data.get('exit_time'):
            if isinstance(data['exit_time'], str):
                exit_time = datetime.fromisoformat(data['exit_time'])
            else:
                exit_time = data['exit_time']

        return cls(
            id=data.get('id'),
            customer_id=data.get('customer_id'),
            entry_time=entry_time,
            exit_time=exit_time,
            dwell_time_seconds=data.get('dwell_time_seconds', 0),
            camera_id=data.get('camera_id')
        )

    def is_active(self) -> bool:
        """Check if visit is still active (not exited)"""
        return self.exit_time is None

    def calculate_dwell_time(self) -> int:
        """Calculate dwell time in seconds"""
        if self.entry_time is None:
            return 0

        end_time = self.exit_time if self.exit_time else datetime.now()
        delta = end_time - self.entry_time
        return int(delta.total_seconds())


# Event type constants
class EventType:
    """Event type constants"""
    ENTRY = 'entry'
    EXIT = 'exit'
    RECOGNIZED = 'recognized'
    UNKNOWN = 'unknown'
    VIP_DETECTED = 'vip_detected'
    BLACKLIST = 'blacklist'
    NEW_CUSTOMER = 'new_customer'
    LONG_DWELL = 'long_dwell'
    REGULAR_VISIT = 'regular_visit'  # Regular customer visit


# Customer segment constants
class CustomerSegment:
    """Customer segment constants"""
    REGULAR = 'regular'
    VIP = 'vip'
    NEW = 'new'
    BLACKLIST = 'blacklist'
