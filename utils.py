#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Recognition Utils - All helper classes and functions
Module độc lập cho face recognition system
"""

import logging as log
import os
import os.path as osp
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from openvino import AsyncInferQueue, PartialShape


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def resize_image(image, size):
    """
    Resize image to target size

    Args:
        image: Input image (HWC format)
        size: Target size (width, height)

    Returns:
        Resized image
    """
    return cv2.resize(image, size)


def crop(frame, roi):
    """
    Crop frame theo ROI

    Args:
        frame: Input frame
        roi: ROI object với position và size

    Returns:
        Cropped frame
    """
    p1 = roi.position.astype(int)
    p1 = np.clip(p1, [0, 0], [frame.shape[1], frame.shape[0]])
    p2 = (roi.position + roi.size).astype(int)
    p2 = np.clip(p2, [0, 0], [frame.shape[1], frame.shape[0]])
    return frame[p1[1]:p2[1], p1[0]:p2[0]]


def cut_rois(frame, rois):
    """
    Crop nhiều ROIs từ frame

    Args:
        frame: Input frame
        rois: List of ROI objects

    Returns:
        List of cropped frames
    """
    return [crop(frame, roi) for roi in rois]


def resize_input(image, target_shape, nchw_layout):
    """
    Resize và format image cho model input

    Args:
        image: Input image (HWC format)
        target_shape: Target shape (N, C, H, W) hoặc (N, H, W, C)
        nchw_layout: True nếu layout là NCHW, False nếu NHWC

    Returns:
        Resized and formatted image
    """
    if nchw_layout:
        _, _, h, w = target_shape
    else:
        _, h, w, _ = target_shape

    resized_image = resize_image(image, (w, h))

    if nchw_layout:
        resized_image = resized_image.transpose((2, 0, 1))  # HWC->CHW

    resized_image = resized_image.reshape(target_shape)
    return resized_image


# ============================================================================
# BASE MODULE CLASS
# ============================================================================

class Module:
    """Base module cho inference với OpenVINO"""

    def __init__(self, core, model_path, model_type):
        self.core = core
        self.model_type = model_type
        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)
        self.model_path = model_path
        self.active_requests = 0
        self.clear()

    def deploy(self, device, max_requests=1):
        """Deploy model lên device"""
        self.max_requests = max_requests
        compiled_model = self.core.compile_model(self.model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_queue = AsyncInferQueue(compiled_model, self.max_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(
            self.model_type, self.model_path, device))

    def completion_callback(self, infer_request, id):
        """Callback khi inference hoàn thành"""
        self.outputs[id] = infer_request.results[self.output_tensor]

    def enqueue(self, input):
        """Đưa request vào queue"""
        if self.max_requests <= self.active_requests:
            log.warning('Processing request rejected - too many requests')
            return False

        self.infer_queue.start_async(input, self.active_requests)
        self.active_requests += 1
        return True

    def wait(self):
        """Đợi tất cả requests hoàn thành"""
        if self.active_requests <= 0:
            return
        self.infer_queue.wait_all()
        self.active_requests = 0

    def get_outputs(self):
        """Lấy outputs sau khi inference"""
        self.wait()
        return [v for _, v in sorted(self.outputs.items())]

    def clear(self):
        """Clear outputs"""
        self.outputs = {}

    def infer(self, inputs):
        """Chạy inference đồng bộ"""
        self.clear()
        self.start_async(*inputs)
        return self.postprocess()


# ============================================================================
# FACE DETECTOR
# ============================================================================

class FaceDetector(Module):
    """Face Detection model wrapper"""

    class Result:
        """Face detection result"""
        OUTPUT_SIZE = 7

        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4]))  # (x, y)
            self.size = np.array((output[5], output[6]))  # (w, h)

        def rescale_roi(self, roi_scale_factor=1.0):
            """Rescale ROI"""
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

        def resize_roi(self, frame_width, frame_height):
            """Resize ROI theo frame size"""
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]
            self.size[1] = self.size[1] * frame_height - self.position[1]

        def clip(self, width, height):
            """Clip ROI trong frame"""
            min_val = [0, 0]
            max_val = [width, height]
            self.position[:] = np.clip(self.position, min_val, max_val)
            self.size[:] = np.clip(self.size, min_val, max_val)

    def __init__(self, core, model, input_size, confidence_threshold=0.5, roi_scale_factor=1.15):
        super(FaceDetector, self).__init__(core, model, 'Face Detection')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        if input_size[0] > 0 and input_size[1] > 0:
            self.model.reshape({self.input_tensor_name: PartialShape([1, 3, *input_size])})
        elif not (input_size[0] == 0 and input_size[1] == 0):
            raise ValueError("Both input height and width should be positive for Face Detector reshape")

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        self.output_shape = self.model.outputs[0].shape
        if len(self.output_shape) != 4 or self.output_shape[3] != self.Result.OUTPUT_SIZE:
            raise RuntimeError("The model expects output shape with {} outputs".format(
                self.Result.OUTPUT_SIZE))

        if confidence_threshold > 1.0 or confidence_threshold < 0:
            raise ValueError("Confidence threshold is expected to be in range [0; 1]")
        if roi_scale_factor < 0.0:
            raise ValueError("Expected positive ROI scale factor")

        self.confidence_threshold = confidence_threshold
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):
        """Preprocess frame"""
        self.input_size = frame.shape
        return resize_input(frame, self.input_shape, self.nchw_layout)

    def start_async(self, frame):
        """Bắt đầu async inference"""
        input_data = self.preprocess(frame)
        self.enqueue(input_data)

    def enqueue(self, input_data):
        """Đưa input vào queue"""
        return super(FaceDetector, self).enqueue({self.input_tensor_name: input_data})

    def postprocess(self):
        """Xử lý outputs"""
        outputs = self.get_outputs()[0]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]

        results = []
        for output in outputs[0][0]:
            result = FaceDetector.Result(output)
            if result.confidence < self.confidence_threshold:
                break  # results are sorted by confidence decrease

            result.resize_roi(self.input_size[1], self.input_size[0])
            result.rescale_roi(self.roi_scale_factor)
            result.clip(self.input_size[1], self.input_size[0])
            results.append(result)

        return results


# ============================================================================
# LANDMARKS DETECTOR
# ============================================================================

class LandmarksDetector(Module):
    """Facial Landmarks Detection model wrapper"""

    POINTS_NUMBER = 5

    def __init__(self, core, model):
        super(LandmarksDetector, self).__init__(core, model, 'Landmarks Detection')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        output_shape = self.model.outputs[0].shape
        if not np.array_equal([1, self.POINTS_NUMBER * 2, 1, 1], output_shape):
            raise RuntimeError("The model expects output shape {}, got {}".format(
                [1, self.POINTS_NUMBER * 2, 1, 1], output_shape))

    def preprocess(self, frame, rois):
        """Preprocess frame và ROIs"""
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input_img, self.input_shape, self.nchw_layout)
                  for input_img in inputs]
        return inputs

    def enqueue(self, input_data):
        """Đưa input vào queue"""
        return super(LandmarksDetector, self).enqueue({self.input_tensor_name: input_data})

    def start_async(self, frame, rois):
        """Bắt đầu async inference"""
        inputs = self.preprocess(frame, rois)
        for input_data in inputs:
            self.enqueue(input_data)

    def postprocess(self):
        """Xử lý outputs"""
        results = [out.reshape((-1, 2)).astype(np.float64) for out in self.get_outputs()]
        return results


# ============================================================================
# FACE IDENTIFIER
# ============================================================================

class FaceIdentifier(Module):
    """Face Re-identification model wrapper"""

    # Reference landmarks for face alignment
    # Taken from intel_models/face-reidentification-retail-0095
    REFERENCE_LANDMARKS = [
        (30.2946 / 96, 51.6963 / 112),  # left eye
        (65.5318 / 96, 51.5014 / 112),  # right eye
        (48.0252 / 96, 71.7366 / 112),  # nose tip
        (33.5493 / 96, 92.3655 / 112),  # left lip corner
        (62.7299 / 96, 92.2041 / 112)   # right lip corner
    ]

    UNKNOWN_ID = -1
    UNKNOWN_ID_LABEL = "Unknown"

    class Result:
        """Identity result"""
        def __init__(self, id, distance, desc):
            self.id = id
            self.distance = distance
            self.descriptor = desc

    def __init__(self, core, model, match_threshold=0.5, match_algo='HUNGARIAN'):
        super(FaceIdentifier, self).__init__(core, model, 'Face Reidentification')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        output_shape = self.model.outputs[0].shape
        if len(output_shape) not in (2, 4):
            raise RuntimeError("The model expects output shape [1, n, 1, 1] or [1, n], got {}".format(
                output_shape))

        self.faces_database = None
        self.match_threshold = match_threshold
        self.match_algo = match_algo

    def set_faces_database(self, database):
        """Set faces database"""
        self.faces_database = database

    def get_identity_label(self, id):
        """Get identity label by id"""
        if not self.faces_database or id == self.UNKNOWN_ID:
            return self.UNKNOWN_ID_LABEL
        return self.faces_database[id].label

    def preprocess(self, frame, rois, landmarks):
        """Preprocess frame, ROIs và landmarks"""
        image = frame.copy()
        inputs = cut_rois(image, rois)
        self._align_rois(inputs, landmarks)
        inputs = [resize_input(input_img, self.input_shape, self.nchw_layout)
                  for input_img in inputs]
        return inputs

    def enqueue(self, input_data):
        """Đưa input vào queue"""
        return super(FaceIdentifier, self).enqueue({self.input_tensor_name: input_data})

    def start_async(self, frame, rois, landmarks):
        """Bắt đầu async inference"""
        inputs = self.preprocess(frame, rois, landmarks)
        for input_data in inputs:
            self.enqueue(input_data)

    def get_threshold(self):
        """Get matching threshold"""
        return self.match_threshold

    def postprocess(self):
        """Xử lý outputs"""
        descriptors = self.get_descriptors()

        matches = []
        if len(descriptors) != 0:
            matches = self.faces_database.match_faces(descriptors, self.match_algo)

        results = []
        unknowns_list = []
        for num, match in enumerate(matches):
            id = match[0]
            distance = match[1]
            if self.match_threshold < distance:
                id = self.UNKNOWN_ID
                unknowns_list.append(num)

            results.append(self.Result(id, distance, descriptors[num]))
        return results, unknowns_list

    def get_descriptors(self):
        """Get descriptor vectors"""
        return [out.flatten() for out in self.get_outputs()]

    @staticmethod
    def normalize(array, axis):
        """Normalize array"""
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std

    @staticmethod
    def get_transform(src, dst):
        """Get affine transform matrix"""
        assert np.array_equal(src.shape, dst.shape) and len(src.shape) == 2, \
            '2d input arrays are expected, got {}'.format(src.shape)
        src_col_mean, src_col_std = FaceIdentifier.normalize(src, axis=0)
        dst_col_mean, dst_col_std = FaceIdentifier.normalize(dst, axis=0)

        u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
        r = np.matmul(u, vt).T

        transform = np.empty((2, 3))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:, 2] = dst_col_mean.T - np.matmul(transform[:, 0:2], src_col_mean.T)
        return transform

    def _align_rois(self, face_images, face_landmarks):
        """Align face images theo landmarks"""
        assert len(face_images) == len(face_landmarks), \
            'Input lengths differ, got {} and {}'.format(len(face_images), len(face_landmarks))

        for image, image_landmarks in zip(face_images, face_landmarks):
            scale = np.array((image.shape[1], image.shape[0]))
            desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=float) * scale
            landmarks = image_landmarks * scale

            transform = FaceIdentifier.get_transform(desired_landmarks, landmarks)
            cv2.warpAffine(image, transform, tuple(scale), image, flags=cv2.WARP_INVERSE_MAP)


# ============================================================================
# FACES DATABASE
# ============================================================================

class FacesDatabase:
    """Database quản lý khuôn mặt đã đăng ký"""

    IMAGE_EXTENSIONS = ['jpg', 'png', 'jpeg']

    class Identity:
        """Identity của một người"""
        def __init__(self, label, descriptors):
            self.label = label
            self.descriptors = descriptors

        @staticmethod
        def cosine_dist(x, y):
            """
            Tính cosine distance giữa 2 vectors
            cosine() trả về 1 - cosine_similarity
            cosine_similarity thuộc [-1, 1]
            (1 - cosine_similarity) thuộc [0, 2]
            Scale về [0, 1] bằng cách chia 2
            """
            return cosine(x, y) * 0.5

    def __init__(self, path, face_identifier, landmarks_detector, face_detector=None, no_show=False):
        path = osp.abspath(path)
        self.fg_path = path
        self.no_show = no_show
        paths = []

        if osp.isdir(path):
            # Tìm ảnh trực tiếp trong gallery và trong các folder con
            paths = []
            try:
                for item in os.listdir(path):
                    item_path = osp.join(path, item)
                    if osp.isfile(item_path):
                        # Ảnh trực tiếp trong gallery
                        if item.split('.')[-1].lower() in self.IMAGE_EXTENSIONS:
                            paths.append(item_path)
                    elif osp.isdir(item_path):
                        # Folder con - tìm ảnh bên trong
                        try:
                            for sub_item in os.listdir(item_path):
                                sub_item_path = osp.join(item_path, sub_item)
                                if osp.isfile(sub_item_path) and sub_item.split('.')[-1].lower() in self.IMAGE_EXTENSIONS:
                                    paths.append(sub_item_path)
                        except OSError as e:
                            log.warning(f"Cannot read subdirectory {item_path}: {e}")
                            continue
            except OSError as e:
                log.warning(f"Cannot read gallery directory {path}: {e}")
                paths = []
        else:
            # Gallery folder không tồn tại - tạo folder mới
            log.info(f"Gallery folder does not exist, creating: {path}")
            try:
                os.makedirs(path, exist_ok=True)
                paths = []
            except OSError as e:
                log.error(f"Cannot create gallery folder {path}: {e}")
                paths = []
                log.warning("Wrong face images database path. Expected a "
                           "path to the directory containing %s files, "
                           "but got '%s'" %
                           (" or ".join(self.IMAGE_EXTENSIONS), path))

        if len(paths) == 0:
            log.info("The images database folder has no images or is empty. Database will be empty but models can still be used.")

        self.database = []
        for path in paths:
            # Lấy label từ tên folder (nếu ảnh trong folder con) hoặc từ tên file
            folder_name = osp.basename(osp.dirname(path))
            file_name = osp.basename(path)
            
            # Nếu ảnh trong folder con của gallery, dùng tên folder làm label
            if folder_name != osp.basename(self.fg_path):
                label = folder_name
            else:
                # Ảnh trực tiếp trong gallery - lấy từ tên file
                label = osp.splitext(file_name)[0]
                # Remove suffix like "-0", "-1" từ label
                import re
                match = re.match(r'(.+?)-\d+$', label)
                if match:
                    label = match.group(1)

            image = cv2.imread(path, flags=cv2.IMREAD_COLOR)

            orig_image = image.copy()

            if face_detector:
                rois = face_detector.infer((image,))
                if len(rois) < 1:
                    log.warning("Not found faces on the image '{}'".format(path))
                    continue
            else:
                w, h = image.shape[1], image.shape[0]
                rois = [FaceDetector.Result([0, 0, 0, 0, 0, w, h])]

            for roi in rois:
                r = [roi]
                landmarks = landmarks_detector.infer((image, r))

                face_identifier.start_async(image, r, landmarks)
                descriptor = face_identifier.get_descriptors()[0]

                if face_detector:
                    mm = self.check_if_face_exist(descriptor, face_identifier.get_threshold())
                    if mm >= 0:
                        # Face đã tồn tại, append descriptor
                        self.database[mm].descriptors.append(descriptor)
                        log.debug("Appending descriptor for existing label {}".format(
                            self.database[mm].label))
                    else:
                        # Face mới
                        log.debug("Adding label {} to the gallery".format(label))
                        self.add_item(descriptor, label)
                else:
                    log.debug("Adding label {} to the gallery".format(label))
                    self.add_item(descriptor, label)

    def match_faces(self, descriptors, match_algo='HUNGARIAN'):
        """Match faces với database"""
        database = self.database
        distances = np.empty((len(descriptors), len(database)))

        for i, desc in enumerate(descriptors):
            for j, identity in enumerate(database):
                dist = []
                for id_desc in identity.descriptors:
                    dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
                distances[i][j] = dist[np.argmin(dist)]

        matches = []
        # MIN_DIST: chọn face với khoảng cách nhỏ nhất
        if match_algo == 'MIN_DIST':
            for i in range(len(descriptors)):
                id = np.argmin(distances[i])
                min_dist = distances[i][id]
                matches.append((id, min_dist))
        else:
            # HUNGARIAN: giải bài toán assignment tối ưu
            _, assignments = linear_sum_assignment(distances)
            for i in range(len(descriptors)):
                if len(assignments) <= i:  # assignment failure
                    matches.append((0, 1.0))
                    continue

                id = assignments[i]
                distance = distances[i, id]
                matches.append((id, distance))

        return matches

    def create_new_label(self, path, id):
        """Tạo label mới"""
        while osp.exists(osp.join(path, "face{}.jpg".format(id))):
            id += 1
        return "face{}".format(id)

    def check_if_face_exist(self, desc, threshold):
        """Kiểm tra xem face đã tồn tại chưa"""
        match = -1
        for j, identity in enumerate(self.database):
            dist = []
            for id_desc in identity.descriptors:
                dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
            if dist[np.argmin(dist)] < threshold:
                match = j
                break
        return match

    def check_if_label_exists(self, label):
        """Kiểm tra xem label đã tồn tại chưa"""
        match = -1
        import re
        name = re.split(r'-\d+$', label)
        if not len(name):
            return -1, label
        label = name[0].lower()

        for j, identity in enumerate(self.database):
            if identity.label.lower() == label:
                match = j
                break
        return match, label

    def dump_faces(self, image, desc, name):
        """Lưu face vào disk và thêm vào database
        
        Ảnh sẽ được lưu vào folder riêng cho mỗi khách hàng: gallery/{label}/image-{index}.jpg
        """
        match, label = self.add_item(desc, name)
        
        # Tạo folder riêng cho khách hàng (nếu chưa có)
        customer_folder = osp.join(self.fg_path, label)
        os.makedirs(customer_folder, exist_ok=True)
        
        # Đếm số ảnh hiện có trong folder của khách hàng này
        if match < 0:
            # Khách hàng mới
            image_index = 0
            match = len(self.database) - 1
        else:
            # Khách hàng đã tồn tại - đếm số ảnh hiện có
            existing_images = [f for f in os.listdir(customer_folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_index = len(existing_images)
        
        # Tạo filename trong folder của khách hàng
        filename = osp.join(customer_folder, f"image-{image_index}.jpg")

        log.debug("Dumping image with label {} and path {} on disk.".format(label, filename))
        if osp.exists(filename):
            log.warning("File with the same name already exists at {}. So it won't be stored.".format(filename))
        else:
            cv2.imwrite(filename, image)
        return match

    def add_item(self, desc, label):
        """Thêm item vào database"""
        match = -1
        if not label:
            label = self.create_new_label(self.fg_path, len(self.database))
            log.warning("Trying to store an item without a label. Assigned label {}.".format(label))
        else:
            match, label = self.check_if_label_exists(label)

        if match < 0:
            self.database.append(FacesDatabase.Identity(label, [desc]))
        else:
            self.database[match].descriptors.append(desc)
            log.debug("Appending new descriptor for label {}.".format(label))

        return match, label

    def __getitem__(self, idx):
        """Get item by index"""
        return self.database[idx]

    def __len__(self):
        """Get database size"""
        return len(self.database)
