#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crops Manager for Face Recognition System
Quản lý ảnh crop khuôn mặt
"""

import os
import cv2
import logging as log
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import numpy as np
import json

from database import Database
from models import Crop


class CropsManager:
    """Manager for handling face crop images"""

    def __init__(self, db: Database, crops_folder: str = "./crops"):
        """
        Initialize Crops Manager

        Args:
            db: Database instance
            crops_folder: Root folder for storing crops
        """
        self.db = db
        self.crops_folder = Path(crops_folder)
        self.ensure_folders()

        log.info(f"CropsManager initialized with folder: {self.crops_folder}")

    def ensure_folders(self):
        """Ensure crops folder structure exists"""
        self.crops_folder.mkdir(exist_ok=True)
        log.info(f"Crops folder ready: {self.crops_folder}")

    def get_today_folder(self) -> Path:
        """Get today's folder (YYYY-MM-DD format)"""
        today = datetime.now().strftime("%Y-%m-%d")
        folder = self.crops_folder / today
        folder.mkdir(exist_ok=True)
        return folder

    def save_crop(self, face_image: np.ndarray,
                  customer_name: str,
                  customer_id: Optional[int] = None,
                  event_id: Optional[int] = None,
                  bbox: Optional[tuple] = None,
                  confidence: float = 0.0) -> Optional[int]:
        """
        Save face crop to disk and database

        Args:
            face_image: Face crop image (numpy array)
            customer_name: Name of customer
            customer_id: Optional customer ID
            event_id: Optional event ID
            bbox: Optional bounding box (xmin, ymin, xmax, ymax)
            confidence: Recognition confidence

        Returns:
            Crop ID, or None if failed
        """
        try:
            # Validate image
            if face_image is None or face_image.size == 0:
                log.warning("Empty face image, skipping crop save")
                return None

            # Generate filename
            timestamp = datetime.now().strftime("%H%M%S")
            customer_prefix = customer_name.replace(" ", "_")
            if customer_id:
                filename = f"CUST-{customer_id}_{customer_prefix}_{timestamp}.jpg"
            else:
                filename = f"UNKNOWN_{timestamp}.jpg"

            # Get today's folder
            today_folder = self.get_today_folder()
            file_path = today_folder / filename

            # Save image
            cv2.imwrite(str(file_path), face_image)
            log.info(f"Crop saved: {file_path}")

            # Prepare bbox for database
            bbox_json = None
            if bbox:
                bbox_dict = {
                    'x': int(bbox[0]),
                    'y': int(bbox[1]),
                    'w': int(bbox[2] - bbox[0]),
                    'h': int(bbox[3] - bbox[1])
                }
                bbox_json = json.dumps(bbox_dict)

            # Save to database
            crop_id = self.db.add_crop(
                file_path=str(file_path),
                customer_id=customer_id,
                event_id=event_id,
                bbox=bbox_json,
                confidence=confidence
            )

            log.info(f"Crop record created in DB: ID {crop_id}")
            return crop_id

        except Exception as e:
            log.error(f"Error saving crop: {e}")
            return None

    def get_crop_image(self, crop_id: int) -> Optional[np.ndarray]:
        """
        Load crop image from disk

        Args:
            crop_id: Crop ID

        Returns:
            Image as numpy array, or None if not found
        """
        crop = self.db.get_crop(crop_id)
        if not crop:
            log.warning(f"Crop ID {crop_id} not found in database")
            return None

        file_path = Path(crop.file_path)
        if not file_path.exists():
            log.warning(f"Crop file not found: {file_path}")
            return None

        image = cv2.imread(str(file_path))
        return image

    def get_crops_for_customer(self, customer_id: int, limit: int = 10) -> List[Crop]:
        """
        Get recent crops for a customer

        Args:
            customer_id: Customer ID
            limit: Maximum number of crops to return

        Returns:
            List of Crop objects
        """
        return self.db.get_crops_by_customer(customer_id, limit)

    def get_recent_crops(self, limit: int = 20) -> List[Crop]:
        """
        Get most recent crops across all customers

        Args:
            limit: Maximum number of crops to return

        Returns:
            List of Crop objects
        """
        return self.db.get_recent_crops(limit)

    def get_crop_with_image(self, crop_id: int) -> Optional[Dict[str, Any]]:
        """
        Get crop data with loaded image

        Args:
            crop_id: Crop ID

        Returns:
            Dictionary with crop data and image
        """
        crop = self.db.get_crop(crop_id)
        if not crop:
            return None

        image = self.get_crop_image(crop_id)

        return {
            'crop': crop,
            'image': image,
            'exists': image is not None
        }

    def delete_crop(self, crop_id: int) -> bool:
        """
        Delete crop from disk and database (GDPR compliance)

        Args:
            crop_id: Crop ID

        Returns:
            True if successful
        """
        try:
            crop = self.db.get_crop(crop_id)
            if not crop:
                log.warning(f"Crop ID {crop_id} not found")
                return False

            # Delete file from disk
            file_path = Path(crop.file_path)
            if file_path.exists():
                file_path.unlink()
                log.info(f"Crop file deleted: {file_path}")

            # Delete from database
            # TODO: Add delete_crop method to Database class
            # self.db.delete_crop(crop_id)

            return True

        except Exception as e:
            log.error(f"Error deleting crop: {e}")
            return False

    def cleanup_old_crops(self, days: int = 7):
        """
        Clean up crops older than specified days

        Args:
            days: Number of days to keep
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Scan crops folder
            for date_folder in self.crops_folder.iterdir():
                if not date_folder.is_dir():
                    continue

                try:
                    folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
                    if folder_date < cutoff_date:
                        # Delete entire folder
                        for file in date_folder.iterdir():
                            file.unlink()
                        date_folder.rmdir()
                        log.info(f"Cleaned up old crops folder: {date_folder}")
                except ValueError:
                    # Not a valid date folder, skip
                    continue

        except Exception as e:
            log.error(f"Error cleaning up old crops: {e}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Returns:
            Dictionary with storage stats
        """
        total_size = 0
        total_files = 0

        for root, dirs, files in os.walk(self.crops_folder):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    file_path = Path(root) / file
                    total_size += file_path.stat().st_size
                    total_files += 1

        return {
            'total_files': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'folder_path': str(self.crops_folder)
        }

    def export_crops_for_customer(self, customer_id: int, output_folder: str) -> int:
        """
        Export all crops for a customer to a folder (for GDPR data export)

        Args:
            customer_id: Customer ID
            output_folder: Output folder path

        Returns:
            Number of crops exported
        """
        try:
            output_path = Path(output_folder)
            output_path.mkdir(exist_ok=True)

            crops = self.db.get_crops_by_customer(customer_id, limit=1000)
            exported = 0

            for crop in crops:
                source_path = Path(crop.file_path)
                if source_path.exists():
                    dest_path = output_path / source_path.name
                    import shutil
                    shutil.copy2(source_path, dest_path)
                    exported += 1

            log.info(f"Exported {exported} crops for customer {customer_id}")
            return exported

        except Exception as e:
            log.error(f"Error exporting crops: {e}")
            return 0
