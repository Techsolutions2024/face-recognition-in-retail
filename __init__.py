#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Recognition Module

Module độc lập cho face recognition sử dụng OpenVINO

Usage:
    # As a script
    python facere.py

    # As a module
    from face_module import FaceRecognitionSystem
    from face_module.utils import FaceDetector, LandmarksDetector, FaceIdentifier
"""

from .utils import (
    FaceDetector,
    LandmarksDetector,
    FaceIdentifier,
    FacesDatabase,
    crop,
    cut_rois,
    resize_image,
    resize_input
)

__version__ = '1.0.0'
__author__ = 'Face Module Team'
__all__ = [
    'FaceDetector',
    'LandmarksDetector',
    'FaceIdentifier',
    'FacesDatabase',
    'crop',
    'cut_rois',
    'resize_image',
    'resize_input'
]
