#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Events Manager for Face Recognition System
Quản lý và theo dõi các sự kiện nhận diện
"""

import logging as log
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from database import Database
from models import EventType, CustomerSegment


class EventsManager:
    """Manager for handling face recognition events"""

    def __init__(self, db: Database):
        """
        Initialize Events Manager
        
        VIP Detection Logic with Revisit Time:
        - Tracks last detection time and last event time for each customer
        - Uses cooldown time to avoid continuous captures (3-5 seconds)
        - Uses revisit threshold to determine new visit (3 hours)
        - 1 event = 1 lượt ghé (visit)

        Args:
            db: Database instance
        """
        self.db = db
        self.last_detected = {}  # customer_id -> timestamp (last time this customer was detected)
        self.last_event_time = {}  # customer_id -> timestamp (last time an event was created for this customer)
        self.detection_cooldown = self.db.get_detection_cooldown()  # seconds between captures for same person
        self.revisit_threshold = self.db.get_revisit_threshold()  # hours before considering a new visit
        
        # Track active sessions - một người đang trong camera
        # Format: {customer_id: {'entry_time': datetime, 'last_seen': datetime, 'confidences': [list], 'event_id': int, 'frame_count': int}}
        self.active_sessions = {}
        self.session_timeout = 10.0  # seconds - nếu không thấy trong 10s thì coi như rời camera

        log.info(f"EventsManager initialized with cooldown: {self.detection_cooldown}s, revisit: {self.revisit_threshold}h, session_timeout: {self.session_timeout}s")

    def log_event(self, event_type: str, customer_name: str,
                  customer_id: Optional[int] = None,
                  camera_id: Optional[int] = None,
                  confidence: float = 0.0,
                  metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Log an event to database

        Args:
            event_type: Type of event (entry, exit, recognized, etc.)
            customer_name: Name of customer
            customer_id: Optional customer ID
            camera_id: Optional camera ID
            confidence: Recognition confidence (0-100)
            metadata: Optional metadata dictionary

        Returns:
            Event ID
        """
        import json
        metadata_str = json.dumps(metadata) if metadata else None

        event_id = self.db.add_event(
            event_type=event_type,
            customer_name=customer_name,
            customer_id=customer_id,
            camera_id=camera_id,
            confidence=confidence,
            metadata=metadata_str
        )

        log.info(f"Event logged: {event_type} - {customer_name} (ID: {event_id})")
        return event_id

    def on_face_recognized(self, customer_name: str, customer_id: Optional[int],
                          confidence: float, bbox: tuple, camera_id: int,
                          face_id: str) -> Optional[Dict[str, Any]]:
        """
        Handle recognized face - Session-based tracking
        
        Logic:
        - 1 lần xuất hiện = 1 session = 1 event
        - Track active sessions: người đó đang trong camera hay đã rời đi
        - Chỉ tạo event khi có session mới (entry) hoặc khi session kết thúc (exit)
        - Update event với duration và confidence summary khi session kết thúc
        - Event type theo segment: VIP, NEW, REGULAR, BLACKLIST
        
        Flow:
        1. Người A xuất hiện lần đầu → Tạo session mới → Tạo event (entry)
        2. Người A tiếp tục xuất hiện → Update session (last_seen, confidences)
        3. Người A rời camera (timeout > 10s) → Kết thúc session → Update event với duration/summary
        4. Người A quay lại → Tạo session mới → Tạo event mới

        Args:
            customer_name: Name of recognized customer
            customer_id: Customer ID (if exists in DB)
            confidence: Recognition confidence
            bbox: Bounding box (xmin, ymin, xmax, ymax)
            camera_id: Camera ID
            face_id: Face ID from gallery (e.g., 'John-0')

        Returns:
            Dictionary with event info if new session, None if continuing session
            - 'event_id': Event ID (None if continuing session)
            - 'should_save_crop': Whether to save crop image
            - 'is_new_session': Whether this is a new session
        """
        now = datetime.now()

        # Get or create customer in database
        if customer_id is None:
            # Try to find by face_id
            customer = self.db.get_customer_by_face_id(face_id)
            if customer:
                customer_id = customer.id
            else:
                # Create new customer
                customer_id = self.db.add_customer(
                    face_id=face_id,
                    name=customer_name,
                    segment=CustomerSegment.NEW
                )
                log.info(f"New customer created in DB: {customer_name} (ID: {customer_id})")

        # Get customer details
        customer = self.db.get_customer(customer_id)
        if not customer:
            return None
        
        # Determine event type based on customer segment
        event_type = self._get_event_type_by_segment(customer.segment)
        
        # Check for active session
        active_session = self.active_sessions.get(customer_id)
        
        if active_session is None:
            # NEW SESSION - Người này vừa xuất hiện lần đầu
            # Calculate average confidence from first detection
            avg_confidence = confidence
            
            event_id = self.log_event(
                event_type=event_type,
                customer_name=customer_name,
                customer_id=customer_id,
                camera_id=camera_id,
                confidence=avg_confidence,  # Use first confidence
                metadata={
                    'bbox': bbox,
                    'face_id': face_id,
                    'entry_time': now.isoformat(),
                    'confidences': [confidence],
                    'frame_count': 1
                }
            )
            
            # Create new session
            self.active_sessions[customer_id] = {
                'entry_time': now,
                'last_seen': now,
                'last_crop_time': now,  # Track last time crop was saved
                'confidences': [confidence],
                'event_id': event_id,
                'frame_count': 1
            }
            
            log.info(f"✅ New session started: {customer_name} (Event ID: {event_id}, Type: {event_type})")
            
            return {
                'event_id': event_id,
                'event_type': event_type,
                'customer_id': customer_id,
                'customer_name': customer_name,
                'is_vip': customer.is_vip(),
                'is_blacklist': customer.is_blacklist(),
                'should_save_crop': True,  # Save crop for new session
                'is_new_session': True
            }
        
        else:
            # CONTINUING SESSION - Người này vẫn đang trong camera
            # Check if session timed out (rời camera và quay lại)
            time_since_last_seen = (now - active_session['last_seen']).total_seconds()
            
            if time_since_last_seen > self.session_timeout:
                # Session timeout - người này đã rời camera trước đó và quay lại
                # End previous session and create new one
                self._end_session(customer_id, active_session['last_seen'])  # End at last_seen
                
                # Calculate average confidence for new session
                avg_confidence = confidence
                
                # Create new session
                event_id = self.log_event(
                    event_type=event_type,
                    customer_name=customer_name,
                    customer_id=customer_id,
                    camera_id=camera_id,
                    confidence=avg_confidence,
                    metadata={
                        'bbox': bbox,
                        'face_id': face_id,
                        'entry_time': now.isoformat(),
                        'confidences': [confidence],
                        'frame_count': 1
                    }
                )
                
                self.active_sessions[customer_id] = {
                    'entry_time': now,
                    'last_seen': now,
                    'last_crop_time': now,  # Track last time crop was saved
                    'confidences': [confidence],
                    'event_id': event_id,
                    'frame_count': 1
                }
                
                log.info(f"✅ New session after timeout: {customer_name} (Event ID: {event_id}, Timeout: {time_since_last_seen:.1f}s)")
                
                return {
                    'event_id': event_id,
                    'event_type': event_type,
                    'customer_id': customer_id,
                    'customer_name': customer_name,
                    'is_vip': customer.is_vip(),
                    'is_blacklist': customer.is_blacklist(),
                    'should_save_crop': True,
                    'is_new_session': True
                }
            else:
                # Same session - update tracking (không tạo event mới)
                active_session['last_seen'] = now
                active_session['confidences'].append(confidence)
                active_session['frame_count'] += 1
                
                # Update event metadata periodically (every 10 frames) - async
                if active_session['frame_count'] % 10 == 0:
                    # Update metadata with latest info
                    avg_confidence = sum(active_session['confidences']) / len(active_session['confidences'])
                    try:
                        event = self.db.get_event(active_session['event_id'])
                        if event:
                            metadata = event.metadata or {}
                            metadata.update({
                                'confidences': active_session['confidences'][-50:],  # Keep last 50
                                'frame_count': active_session['frame_count'],
                                'confidence_avg': avg_confidence,
                                'last_seen': now.isoformat()
                            })
                            self.db.update_event_metadata(active_session['event_id'], metadata)
                    except Exception as e:
                        log.error(f"Error updating session metadata: {e}")
                
                # Don't save crop on every frame - use cooldown (time-based)
                should_save_crop = False
                time_since_last_crop = (now - active_session.get('last_crop_time', active_session['entry_time'])).total_seconds()
                if time_since_last_crop >= self.detection_cooldown:
                    should_save_crop = True
                    active_session['last_crop_time'] = now  # Update last crop time
                
                return {
                    'event_id': active_session['event_id'],
                    'event_type': event_type,
                    'customer_id': customer_id,
                    'customer_name': customer_name,
                    'is_vip': customer.is_vip(),
                    'is_blacklist': customer.is_blacklist(),
                    'should_save_crop': should_save_crop,
                    'is_new_session': False  # Continuing session
                }

    def on_unknown_face(self, confidence: float, bbox: tuple, camera_id: int) -> Optional[Dict[str, Any]]:
        """
        Handle unknown face detection - Session-based tracking
        
        Track unknown faces by bbox hash (simple approach) to avoid spam
        Mỗi unknown face xuất hiện = 1 session = 1 event

        Args:
            confidence: Detection confidence
            bbox: Bounding box (xmin, ymin, xmax, ymax)
            camera_id: Camera ID

        Returns:
            Dictionary with event info if new session, None if continuing session
        """
        now = datetime.now()
        
        # Create unique ID for unknown face based on bbox position (simple hash)
        # Same bbox position in same time window = same unknown person
        bbox_hash = hash((int(bbox[0] // 50), int(bbox[1] // 50), camera_id))  # Grid-based hash
        unknown_id = f"unknown_{bbox_hash}"
        
        # Check for active unknown session
        active_session = self.active_sessions.get(unknown_id)
        
        if active_session is None:
            # NEW UNKNOWN SESSION
            event_id = self.log_event(
                event_type=EventType.UNKNOWN,
                customer_name='Unknown',
                customer_id=None,
                camera_id=camera_id,
                confidence=confidence,
                metadata={
                    'bbox': bbox,
                    'entry_time': now.isoformat(),
                    'confidences': [confidence],
                    'frame_count': 1
                }
            )
            
            # Create session for unknown face
            self.active_sessions[unknown_id] = {
                'entry_time': now,
                'last_seen': now,
                'last_crop_time': now,  # Track last time crop was saved
                'last_bbox': bbox,  # Track last bbox to detect movement
                'confidences': [confidence],
                'event_id': event_id,
                'frame_count': 1,
                'is_unknown': True
            }
            
            log.info(f"✅ New unknown session started (Event ID: {event_id})")
            
            return {
                'event_id': event_id,
                'event_type': EventType.UNKNOWN,
                'customer_name': 'Unknown',
                'customer_id': None,
                'is_unknown': True,
                'should_save_crop': True,
                'is_new_session': True
            }
        
        else:
            # CONTINUING UNKNOWN SESSION
            time_since_last_seen = (now - active_session['last_seen']).total_seconds()
            
            if time_since_last_seen > self.session_timeout:
                # Session timeout - unknown face đã rời và quay lại
                self._end_session(unknown_id, active_session['last_seen'])
                
                # Create new session
                event_id = self.log_event(
                    event_type=EventType.UNKNOWN,
                    customer_name='Unknown',
                    customer_id=None,
                    camera_id=camera_id,
                    confidence=confidence,
                    metadata={
                        'bbox': bbox,
                        'entry_time': now.isoformat(),
                        'confidences': [confidence],
                        'frame_count': 1
                    }
                )
                
                self.active_sessions[unknown_id] = {
                    'entry_time': now,
                    'last_seen': now,
                    'last_crop_time': now,  # Track last time crop was saved
                    'last_bbox': bbox,  # Track last bbox to detect movement
                    'confidences': [confidence],
                    'event_id': event_id,
                    'frame_count': 1,
                    'is_unknown': True
                }
                
                log.info(f"✅ New unknown session after timeout (Event ID: {event_id})")
                
                return {
                    'event_id': event_id,
                    'event_type': EventType.UNKNOWN,
                    'customer_name': 'Unknown',
                    'customer_id': None,
                    'is_unknown': True,
                    'should_save_crop': True,
                    'is_new_session': True
                }
            else:
                # Same session - update tracking
                active_session['last_seen'] = now
                active_session['confidences'].append(confidence)
                active_session['frame_count'] += 1
                
                # Update event metadata periodically
                if active_session['frame_count'] % 10 == 0:
                    avg_confidence = sum(active_session['confidences']) / len(active_session['confidences'])
                    try:
                        event = self.db.get_event(active_session['event_id'])
                        if event:
                            metadata = event.metadata or {}
                            metadata.update({
                                'confidences': active_session['confidences'][-50:],
                                'frame_count': active_session['frame_count'],
                                'confidence_avg': avg_confidence,
                                'last_seen': now.isoformat()
                            })
                            self.db.update_event_metadata(active_session['event_id'], metadata)
                    except Exception as e:
                        log.error(f"Error updating unknown session metadata: {e}")
                
                # Chiến lược chụp ảnh cho khách lạ:
                # - Chụp 1 ảnh khi xuất hiện lần đầu (đã xử lý ở trên)
                # - Chụp thêm ảnh chỉ khi có sự thay đổi đủ lớn:
                #   + Di chuyển đáng kể (bbox center di chuyển > 50 pixels)
                #   + Hoặc đã qua 10 giây kể từ lần chụp cuối
                should_save_crop = False
                
                # Kiểm tra di chuyển bbox (sự thay đổi vị trí)
                last_bbox = active_session.get('last_bbox')
                if last_bbox:
                    # Tính center của bbox cũ và mới
                    old_center_x = (last_bbox[0] + last_bbox[2]) / 2
                    old_center_y = (last_bbox[1] + last_bbox[3]) / 2
                    new_center_x = (bbox[0] + bbox[2]) / 2
                    new_center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Khoảng cách di chuyển
                    movement_distance = ((new_center_x - old_center_x) ** 2 + 
                                       (new_center_y - old_center_y) ** 2) ** 0.5
                    
                    # Nếu di chuyển > 50 pixels → chụp ảnh (góc nhìn hoặc vị trí thay đổi đáng kể)
                    if movement_distance > 50:
                        should_save_crop = True
                        active_session['last_bbox'] = bbox  # Update last bbox
                        active_session['last_crop_time'] = now  # Update last crop time
                        log.info(f"📸 Unknown face moved significantly ({movement_distance:.1f}px) - saving crop")
                
                # Nếu chưa chụp do di chuyển, kiểm tra thời gian (ít nhất 10 giây)
                if not should_save_crop:
                    time_since_last_crop = (now - active_session.get('last_crop_time', active_session['entry_time'])).total_seconds()
                    if time_since_last_crop >= 10.0:  # 10 giây
                        should_save_crop = True
                        active_session['last_crop_time'] = now  # Update last crop time
                        active_session['last_bbox'] = bbox  # Update last bbox
                        log.info(f"📸 Unknown face - time threshold reached ({time_since_last_crop:.1f}s) - saving crop")
                
                # Update last bbox nếu chưa update
                if not should_save_crop:
                    active_session['last_bbox'] = bbox
                
                # Don't create new event - continuing session
                return {
                    'event_id': active_session['event_id'],
                    'event_type': EventType.UNKNOWN,
                    'customer_name': 'Unknown',
                    'customer_id': None,
                    'is_unknown': True,
                    'should_save_crop': should_save_crop,  # Chụp nếu có thay đổi đủ lớn
                    'is_new_session': False
                }

    def update_cooldown(self):
        """Update cooldown time from database (called when admin changes settings)"""
        self.detection_cooldown = self.db.get_detection_cooldown()
        log.info(f"Detection cooldown updated to {self.detection_cooldown}s")

    def update_revisit_threshold(self):
        """Update revisit threshold from database (called when admin changes settings)"""
        self.revisit_threshold = self.db.get_revisit_threshold()
        log.info(f"Revisit threshold updated to {self.revisit_threshold}h")

    def get_active_customers(self) -> int:
        """Get count of recently detected customers (within cooldown window)"""
        now = datetime.now()
        active_count = 0
        for customer_id, last_time in self.last_detected.items():
            time_elapsed = (now - last_time).total_seconds()
            if time_elapsed < self.detection_cooldown * 2:  # Within 2x cooldown window
                active_count += 1
        return active_count

    def clear_detection_history(self):
        """Clear detection history (e.g., at end of day)"""
        self.last_detected.clear()
        self.last_event_time.clear()
        # End all active sessions before clearing
        now = datetime.now()
        for customer_id in list(self.active_sessions.keys()):
            self._end_session(customer_id, now)
        log.info("Detection history cleared")

    def _end_session(self, customer_id: int, end_time: datetime):
        """End an active session and update event with duration and confidence summary"""
        session = self.active_sessions.get(customer_id)
        if not session:
            return
        
        # Calculate duration
        duration_seconds = (end_time - session['entry_time']).total_seconds()
        duration_formatted = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
        
        # Calculate confidence summary
        confidences = session['confidences']
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)
        else:
            avg_confidence = max_confidence = min_confidence = 0.0
        
        # Update event metadata
        event = self.db.get_event(session['event_id'])
        if event:
            import json
            metadata = event.metadata or {}
            metadata.update({
                'exit_time': end_time.isoformat(),
                'duration_seconds': duration_seconds,
                'duration_formatted': duration_formatted,
                'frame_count': session['frame_count'],
                'confidence_avg': avg_confidence,
                'confidence_max': max_confidence,
                'confidence_min': min_confidence,
                'confidence_count': len(confidences)
            })
            
            # Update event in database
            self.db.update_event_metadata(session['event_id'], metadata)
            log.debug(f"Session ended for customer {customer_id}: duration={duration_formatted}, avg_confidence={avg_confidence:.1f}%")
        
        # Remove session
        del self.active_sessions[customer_id]

    def _get_event_type_by_segment(self, segment: str) -> str:
        """Get event type based on customer segment"""
        if segment == CustomerSegment.VIP:
            return EventType.VIP_DETECTED
        elif segment == CustomerSegment.NEW:
            return EventType.NEW_CUSTOMER
        elif segment == CustomerSegment.BLACKLIST:
            return EventType.BLACKLIST
        else:
            return EventType.REGULAR_VISIT  # Regular customer visit

    def check_timeout_sessions(self):
        """Check and end sessions that have timed out"""
        now = datetime.now()
        timeout_customers = []
        
        for customer_id, session in list(self.active_sessions.items()):
            time_since_last_seen = (now - session['last_seen']).total_seconds()
            if time_since_last_seen > self.session_timeout:
                timeout_customers.append(customer_id)
        
        # End timeout sessions
        for customer_id in timeout_customers:
            session = self.active_sessions.get(customer_id)
            if session:
                self._end_session(customer_id, session['last_seen'])


    def get_stats_today(self) -> Dict[str, Any]:
        """
        Get statistics for today

        Returns:
            Dictionary with statistics
        """
        event_count = self.db.get_event_count_today()
        event_stats = self.db.get_event_stats()

        return {
            'total_events': event_count,
            'event_breakdown': event_stats,
            'recently_detected': self.get_active_customers(),
            'cooldown_time': self.detection_cooldown,
            'revisit_threshold': self.revisit_threshold
        }
