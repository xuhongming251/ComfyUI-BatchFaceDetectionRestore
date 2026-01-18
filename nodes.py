# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import onnxruntime
import warnings
from typing import List, Tuple
import mediapipe as mp
from PIL import Image
from functools import reduce

script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

# ==================== 工具类和函数 ====================

def box_convert_simple(box, convert_type='xyxy2xywh'):
    if convert_type == 'xyxy2xywh':
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    elif convert_type == 'xywh2xyxy':
        return [box[0], box[1], box[2] + box[0], box[3] + box[1]]
    elif convert_type == 'xyxy2ctwh':
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]]
    elif convert_type == 'ctwh2xyxy':
        return [box[0] - box[2] // 2, box[1] - box[3] // 2, box[0] + (box[2] - box[2] // 2), box[1] + (box[3] - box[3] // 2)]

def bbox_from_detector(bbox, input_resolution=(224, 224), rescale=1.25):
    CROP_IMG_HEIGHT, CROP_IMG_WIDTH = input_resolution
    CROP_ASPECT_RATIO = CROP_IMG_HEIGHT / float(CROP_IMG_WIDTH)
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = np.array([center_x, center_y])
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * CROP_ASPECT_RATIO, bbox_h)
    scale = np.array([bbox_size / CROP_ASPECT_RATIO, bbox_size]) / 200.0
    scale *= rescale
    return center, scale

def get_transform(center, scale, res, rot=0):
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1

def crop(img, center, scale, res):
    ul = np.array(transform([1, 1], center, max(scale), res, invert=1)) - 1
    br = np.array(transform([res[1] + 1, res[0] + 1], center, max(scale), res, invert=1)) - 1
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)
    new_img = cv2.resize(new_img, (res[1], res[0]))
    return new_img, new_shape, (old_x, old_y), (new_x, new_y)

def _get_max_preds(heatmaps):
    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals

def transform_preds(coords, center, scale, output_size, use_udp=False):
    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]
    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target_coords

def _gaussian_blur(heatmaps, kernel=11):
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps

def _taylor(heatmap, coord):
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] - heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] + heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord

def keypoints_from_heatmaps(heatmaps, center, scale, unbiased=False, post_process='default', kernel=11, use_udp=False):
    heatmaps = heatmaps.copy()
    if post_process == 'default' and unbiased:
        post_process = 'unbiased'
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)
    N, K, H, W = heatmaps.shape
    preds, maxvals = _get_max_preds(heatmaps)
    if post_process == 'unbiased':
        heatmaps = np.log(np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
        for n in range(N):
            for k in range(K):
                preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
    elif post_process is not None:
        for n in range(N):
            for k in range(K):
                heatmap = heatmaps[n][k]
                px, py = int(preds[n][k][0]), int(preds[n][k][1])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    diff = np.array([heatmap[py][px + 1] - heatmap[py][px - 1], heatmap[py + 1][px] - heatmap[py - 1][px]])
                    preds[n][k] += np.sign(diff) * .25
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H], use_udp=use_udp)
    return preds, maxvals

def split_kp2ds_for_aa(kp2ds, ret_face=False):
    kp2ds_body = (kp2ds[[0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]] + kp2ds[[0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]]) / 2
    kp2ds_lhand = kp2ds[91:112]
    kp2ds_rhand = kp2ds[112:133]
    kp2ds_face = kp2ds[22:91]
    if ret_face:
        return kp2ds_body.copy(), kp2ds_lhand.copy(), kp2ds_rhand.copy(), kp2ds_face.copy()
    return kp2ds_body.copy(), kp2ds_lhand.copy(), kp2ds_rhand.copy()

def load_pose_metas_from_kp2ds_seq(kp2ds_seq, width, height):
    metas = []
    last_kp2ds_body = None
    for kps in kp2ds_seq:
        kps = kps.copy()
        kps[:, 0] /= width
        kps[:, 1] /= height
        kp2ds_body, kp2ds_lhand, kp2ds_rhand, kp2ds_face = split_kp2ds_for_aa(kps, ret_face=True)
        if last_kp2ds_body is not None and kp2ds_body[:, :2].min(axis=1).max() < 0:
            kp2ds_body = last_kp2ds_body
        last_kp2ds_body = kp2ds_body
        meta = {"width": width, "height": height, "keypoints_body": kp2ds_body, "keypoints_left_hand": kp2ds_lhand, "keypoints_right_hand": kp2ds_rhand, "keypoints_face": kp2ds_face}
        metas.append(meta)
    return metas

def get_face_bboxes(kp2ds, scale, image_shape):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[1:] * (w, h)
    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)
    initial_width, initial_height = max_x - min_x, max_y - min_y
    initial_area = initial_width * initial_height
    expanded_area = initial_area * scale
    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))
    delta_width, delta_height = (new_width - initial_width) / 2, (new_height - initial_height) / 4
    expanded_min_x, expanded_max_x = max(min_x - delta_width, 0), min(max_x + delta_width, w)
    expanded_min_y, expanded_max_y = max(min_y - 3 * delta_height, 0), min(max_y + delta_height, h)
    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]

# ==================== MediaPipe 辅助类 ====================

def get_mediapipe_model_path(model_type="segmenter"):
    model_folder_path = os.path.join(folder_paths.models_dir, "mediapipe")
    if model_type == "segmenter":
        model_name = "selfie_multiclass_256x256.tflite"
        model_url = f"https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/{model_name}"
    else:
        model_name = "face_landmarker.task"
        model_url = f"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float32/latest/{model_name}"
    
    model_file_path = os.path.join(model_folder_path, model_name)
    
    if not os.path.exists(model_file_path):
        os.makedirs(model_folder_path, exist_ok=True)
        try:
            import wget
            print(f"Downloading '{model_name}' model to {model_file_path}")
            wget.download(model_url, model_file_path)
        except Exception:
            import urllib.request
            print(f"Downloading '{model_name}' via urllib...")
            urllib.request.urlretrieve(model_url, model_file_path)
    return model_file_path

class MediaPipeDetector:
    def __init__(self):
        # 1. 初始化分割模型 (用于高质量蒙版生成)
        seg_model_path = get_mediapipe_model_path("segmenter")
        with open(seg_model_path, "rb") as f:
            seg_buffer = f.read()
        self.seg_options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=seg_buffer),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(self.seg_options)

        # 2. 初始化 468 点 Face Mesh 模型 (用于精确 BBox)
        mesh_model_path = get_mediapipe_model_path("face_mesh")
        with open(mesh_model_path, "rb") as f:
            mesh_buffer = f.read()
        self.mesh_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=mesh_buffer),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3, # 降低门槛以提高覆盖
            min_face_presence_confidence=0.3,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(self.mesh_options)

    def get_face_data(self, image_np):
        """
        获取 468 点关键点、BBox 以及 语义分割掩码
        """
        h, w, _ = image_np.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        
        # 同时运行 Mesh 和 Segmenter
        mesh_result = self.landmarker.detect(mp_image)
        seg_result = self.segmenter.segment(mp_image)
        
        # 提取语义分割的人脸皮肤掩码 (Category 3)
        face_skin_mask = seg_result.confidence_masks[3].numpy_view()
        
        m_x1, m_y1, m_x2, m_y2 = float('inf'), float('inf'), float('-inf'), float('-inf')
        mesh_coords = None
        if mesh_result.face_landmarks:
            landmarks = mesh_result.face_landmarks[0]
            mesh_coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
            m_x1, m_y1 = mesh_coords.min(axis=0)
            m_x2, m_y2 = mesh_coords.max(axis=0)

        s_x1, s_y1, s_x2, s_y2 = float('inf'), float('inf'), float('-inf'), float('-inf')
        if np.any(face_skin_mask > 0.4):
            y_coords, x_coords = np.where(face_skin_mask > 0.4)
            s_x1, s_y1, s_x2, s_y2 = x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()
        
        # 取并集，得到人脸的最精细边界
        x1 = min(m_x1, s_x1)
        y1 = min(m_y1, s_y1)
        x2 = max(m_x2, s_x2)
        y2 = max(m_y2, s_y2)

        if x1 == float('inf'):
            return None, None, face_skin_mask
        
        # 返回 [x1, y1, x2, y2, score]
        bbox = np.array([x1, y1, x2, y2, 1.0])
        return mesh_coords, bbox, face_skin_mask

    def close(self):
        if hasattr(self, 'segmenter'): self.segmenter.close()
        if hasattr(self, 'landmarker'): self.landmarker.close()

# Core landmarks for pose visualization (from DreamID-V)
CORE_LANDMARK_INDICES = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    1, 2, 5, 6, 48, 64, 94, 98, 168, 195, 197, 278, 294, 324, 327, 4, 24,
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    468, 473, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 70, 63, 105, 66, 107,
    300, 293, 334, 296, 336, 156,
]

# Face oval landmarks for mask generation (from DreamID-V)
FACE_OVAL_INDICES = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
]

class FaceMeshAlign:
    """Align pose landmarks from video to reference image"""
    
    def __call__(self, video_landmarks_list: List[np.ndarray], 
                ref_landmarks: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Align video landmarks to reference image landmarks
        
        Args:
            video_landmarks_list: List of landmark arrays for each frame
            ref_landmarks: Reference image landmarks [478, 2]
            
        Returns:
            success: Whether alignment succeeded
            aligned_landmarks: Aligned landmarks [num_frames, 478, 2]
        """
        if not video_landmarks_list or ref_landmarks is None:
            return False, None
        
        # Determine the number of landmarks from the reference
        num_landmarks = ref_landmarks.shape[0]
        
        num_frames = len(video_landmarks_list)
        aligned = np.zeros((num_frames, num_landmarks, 2), dtype=np.float32)
        
        for i, frame_lm in enumerate(video_landmarks_list):
            if frame_lm is not None:
                # If the frame landmarks have more/less points, we might need to handle it.
                # For now, we assume they are compatible or we just take what we have.
                take_pts = min(num_landmarks, frame_lm.shape[0])
                aligned[i, :take_pts] = frame_lm[:take_pts, :2]
                if take_pts < num_landmarks:
                    # Fill remaining with reference if missing (e.g. iris points)
                    aligned[i, take_pts:] = ref_landmarks[take_pts:, :2]
            else:
                # If detection failed, use previous frame or reference
                if i > 0:
                    aligned[i] = aligned[i-1]
                else:
                    aligned[i] = ref_landmarks[:, :2]
        
        return True, aligned

class SimpleOnnxInference(object):
    def __init__(self, checkpoint, device='CUDAExecutionProvider', **kwargs):
        self.checkpoint = checkpoint
        self.init_kwargs = kwargs
        provider = [device, 'CPUExecutionProvider'] if device == 'CUDAExecutionProvider' else [device]
        self.provider = provider
        self.session = onnxruntime.InferenceSession(checkpoint, providers=provider)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_resolution = np.array(self.session.get_inputs()[0].shape[2:])

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def cleanup(self):
        if hasattr(self, 'session') and self.session is not None:
            del self.session
            self.session = None

    def reinit(self, provider=None):
        if provider is not None:
            self.provider = provider
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.checkpoint, providers=self.provider)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_resolution = np.array(self.session.get_inputs()[0].shape[2:])

class Yolo(SimpleOnnxInference):
    def __init__(self, checkpoint, device='cuda', threshold_conf=0.05, threshold_iou=0.5, cat_id=[1], **kwargs):
        super(Yolo, self).__init__(checkpoint, device=device, **kwargs)
        self.input_width, self.input_height = 640, 640
        self.threshold_conf, self.threshold_iou, self.cat_id = threshold_conf, threshold_iou, cat_id

    def postprocess(self, output, shape_raw):
        outputs = np.squeeze(output)
        if len(outputs.shape) == 1: outputs = outputs[None]
        if output.shape[-1] != 6 and output.shape[1] == 84: outputs = np.transpose(outputs)
        x_factor, y_factor = shape_raw[1] / self.input_width, shape_raw[0] / self.input_height
        boxes, scores, class_ids = [], [], []
        if outputs.shape[-1] == 6:
            max_scores = outputs[:, 4]
            classid = outputs[:, -1]
            mask = (max_scores >= self.threshold_conf)
            max_scores, classid, boxes = max_scores[mask], classid[mask], outputs[:, :4][mask]
            boxes[:, [0, 2]] *= x_factor
            boxes[:, [1, 3]] *= y_factor
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
        else:
            classes_scores = outputs[:, 4:]
            max_scores = np.amax(classes_scores, -1)
            mask = max_scores >= self.threshold_conf
            classid = np.argmax(classes_scores[mask], -1)
            xywh = outputs[:, :4][mask]
            left, top = (xywh[:, 0] - xywh[:, 2] / 2) * x_factor, (xywh[:, 1] - xywh[:, 3] / 2) * y_factor
            width, height = xywh[:, 2] * x_factor, xywh[:, 3] * y_factor
            boxes = np.stack([left, top, width, height], axis=-1)
        boxes, scores, class_ids = boxes.astype(np.int32).tolist(), max_scores.tolist(), classid.tolist()
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold_conf, self.threshold_iou)
        results = []
        for i in indices:
            results.append(box_convert_simple(boxes[i], 'xywh2xyxy') + [scores[i], class_ids[i]])
        return np.array(results)

    def forward(self, img, shape_raw, **kwargs):
        if isinstance(img, torch.Tensor):
            img, shape_raw = img.cpu().numpy(), shape_raw.cpu().numpy()
        outputs = self.session.run(None, {self.input_name: img})[0]
        person_results = []
        for i in range(len(outputs)):
            res = self.postprocess(outputs[i], shape_raw[i])
            if len(res) > 0:
                areas = (res[:, 2] - res[:, 0]) * (res[:, 3] - res[:, 1])
                max_idx = np.argmax(areas)
                person_results.append([{'bbox': res[max_idx, :5], 'track_id': 0}])
            else:
                person_results.append([{'bbox': np.array([0., 0., shape_raw[i][1], shape_raw[i][0], -1]), 'track_id': -1}])
        return person_results

class ViTPose(SimpleOnnxInference):
    def forward(self, img, center, scale, **kwargs):
        heatmaps = self.session.run([], {self.input_name: img})[0]
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=center, scale=scale*200, unbiased=True, use_udp=False)
        return np.concatenate([points, prob], axis=2)

from comfy import model_management as mm
from comfy.utils import ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))



# ==================== 模型加载器 ====================

class BatchFaceDetectionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "yolo_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "onnx_device": (["CUDAExecutionProvider", "CPUExecutionProvider"], {"default": "CUDAExecutionProvider", "tooltip": "Device to run the ONNX models on"}),
            },
        }
    
    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FaceDetectionRestore"
    
    def loadmodel(self, vitpose_model, yolo_model, onnx_device):
        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)
        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)
        model = {
            "vitpose": vitpose,
            "yolo": yolo,
        }
        return (model, )

# ==================== 节点类 ====================
class BatchFaceDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "output_width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 1, "tooltip": "output resolution"}),
                "output_heigh": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 1, "tooltip": "output resolution"}),
                "min_face_size": ("INT", {"default": 30, "min": 0, "max": 1024, "step": 1, "tooltip": "Ignore faces smaller than this size (pixels)"}),
                "face_margin_left": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1, "tooltip": ""}),
                "face_margin_top": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1, "tooltip": ""}),
                "face_margin_right": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1, "tooltip": ""}),
                "face_margin_bottom": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1, "tooltip": ""}),
                "output_pose_video": ("BOOLEAN", {"default": False}),
                "output_face_pose": ("BOOLEAN", {"default": False}),
                "mask_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold for mask binarization"}),
                "hold_max_frame": ("INT", {"default": -1, "min": -1, "max": 99999, "step": 1, "tooltip": "Maximum frames to hold the mask after last detection. -1 means infinite hold."}),
                "hold_backward": ("BOOLEAN", {"default": False, "tooltip": "Whether to also hold the mask backwards from detection (support no-face to face transitions)."}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "BBOX", "IMAGE", "IMAGE")
    RETURN_NAMES = ("face_images", "face_masks", "face_masks_hold", "face_masks_hold_all", "face_bboxes", "images_pose", "face_images_pose")
    FUNCTION = "process"
    CATEGORY = "BatchFaceDetection"
    def process(self, model, images, output_width = 512, output_heigh=512, min_face_size=30, face_margin_left=0, face_margin_top=0,face_margin_right=0,face_margin_bottom=0, output_pose_video=False, output_face_pose=False, mask_threshold=0.5, hold_max_frame=-1, hold_backward=False, ref_image=None):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()
        images_uint8 = (images_np * 255).clip(0, 255).astype(np.uint8)

        detector.reinit()
        pose_model.reinit()
        mp_detector = MediaPipeDetector()

        total_steps = B * 2
        if output_pose_video or output_face_pose:
            total_steps += B
        comfy_pbar = ProgressBar(total_steps)
        
        face_images = []
        face_masks = []
        face_masks_hold = []
        face_masks_hold_all = []
        face_bboxes_out = []
        
        mp_count = 0
        fallback_count = 0

        # 第一阶段：全序列检测与数据收集
        raw_detections = []

        for i in tqdm(range(B), desc="Detecting Faces"):
            img = images_np[i]
            img_u8 = images_uint8[i]
            
            # 尝试 MediaPipe (468点)
            mesh_pts, mesh_bbox, seg_mask = mp_detector.get_face_data(img_u8)
            
            is_valid_face = False
            # 检查 MediaPipe 检测结果的大小
            if mesh_bbox is not None:
                mx1, my1, mx2, my2 = mesh_bbox[:4]
                if (mx2 - mx1) >= min_face_size and (my2 - my1) >= min_face_size:
                    raw_detections.append({
                        "mesh_pts": mesh_pts,
                        "mesh_bbox": mesh_bbox,
                        "seg_mask": seg_mask,
                        "type": "mp",
                        "is_real": True
                    })
                    mp_count += 1
                    is_valid_face = True
            
            if not is_valid_face:
                # 尝试 YOLO + ViTPose (68点)
                yolo_res = detector(cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None], shape)[0][0]
                bbox = yolo_res["bbox"]
                if bbox is not None and bbox[-1] > 0:
                    # 根据 detector 实现，bbox 可能是 [x1, y1, x2, y2, score, ...]
                    bx1, by1, bx2, by2 = bbox[:4]
                    if (bx2 - bx1) >= min_face_size and (by2 - by1) >= min_face_size:
                        center, scale = bbox_from_detector(bbox, (256, 192), rescale=1.25)
                        img_crop_tmp = crop(img, center, scale, (256, 192))[0]
                        img_norm = (img_crop_tmp - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                        img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
                        all_kps = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])[0]
                        
                        raw_detections.append({
                            "kps": all_kps[22:91, :2],
                            "type": "yolo",
                            "is_real": True
                        })
                        fallback_count += 1
                        is_valid_face = True
            
            if not is_valid_face:
                raw_detections.append(None) # 标记为检测失败
            
            comfy_pbar.update(1)

        # 第二阶段：双向传播处理缺失帧 (Adaption for missing faces)
        # 1. 向后传播 (使用后面第一帧有效数据填充前面)
        first_valid_idx = -1
        for i in range(B):
            if raw_detections[i] is not None:
                first_valid_idx = i
                break
        
        if first_valid_idx > 0:
            for i in range(first_valid_idx):
                raw_detections[i] = raw_detections[first_valid_idx].copy()
                raw_detections[i]["is_real"] = False
                
        # 2. 向前传播 (使用前面最后一帧有效数据填充后面)
        last_valid_data = None
        for i in range(B):
            if raw_detections[i] is not None:
                last_valid_data = raw_detections[i]
            else:
                if last_valid_data is not None:
                    raw_detections[i] = last_valid_data.copy()
                    raw_detections[i]["is_real"] = False
                else:
                    raw_detections[i] = None

        # 第三阶段：执行裁剪与 Mask 生成 (引入时间轴平滑以减少抖动)
        # 3.1 收集原始裁剪参数
        raw_centers = np.zeros((B, 2))
        raw_sizes = np.zeros((B, 2))
        is_real_list = np.zeros(B, dtype=bool)

        for i in range(B):
            data = raw_detections[i]
            if data is not None:
                if data["type"] == "mp":
                    x1, y1, x2, y2 = data["mesh_bbox"][:4]
                else:
                    tx1, tx2, ty1, ty2 = get_face_bboxes(data["kps"], scale=1.3, image_shape=(H, W))
                    x1, y1, x2, y2 = tx1, ty1, tx2, ty2
                
                raw_centers[i] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                raw_sizes[i] = [x2 - x1, y2 - y1]
                is_real_list[i] = data.get("is_real", False)

        # 3.2 坐标层面的线性插值 (填充缺失帧，比直接复制邻帧更平滑)
        real_indices = np.where(is_real_list)[0]
        if len(real_indices) == 0:
            raw_centers[:] = [W / 2.0, H / 2.0]
            raw_sizes[:] = [min(W, H) * 0.5, min(W, H) * 0.5]
        else:
            for dim in range(2):
                raw_centers[:, dim] = np.interp(np.arange(B), real_indices, raw_centers[real_indices, dim])
                raw_sizes[:, dim] = np.interp(np.arange(B), real_indices, raw_sizes[real_indices, dim])

        # 3.3 时间轴平滑滤波 (均值滤波)
        def moving_average(data, window):
            if B <= window: return data
            smoothed = np.zeros_like(data)
            half = window // 2
            for i in range(B):
                start = max(0, i - half)
                end = min(B, i + half + 1)
                smoothed[i] = np.mean(data[start:end], axis=0)
            return smoothed

        # 使用较宽的窗口进行中心点平滑，以彻底消除检测抖动
        smooth_centers = moving_average(raw_centers, 9)
        smooth_sizes = moving_average(raw_sizes, 7)

        final_geom_data = [None] * B
        target_aspect = output_width / (output_heigh + 1e-6)

        for i in range(B):
            cx, cy = smooth_centers[i]
            sw, sh = smooth_sizes[i]

            # 保持目标比例
            if sw / (sh + 1e-6) > target_aspect:
                sh = sw / target_aspect
            else:
                sw = sh * target_aspect

            # 应用 Margin
            x1 = cx - (sw / 2.0) - face_margin_left
            x2 = cx + (sw / 2.0) + face_margin_right
            y1 = cy - (sh / 2.0) - face_margin_top
            y2 = cy + (sh / 2.0) + face_margin_bottom

            # 转换为整数并进行边界检查
            x1_c, y1_c, x2_c, y2_c = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))
            
            data = raw_detections[i]
            if x2_c > x1_c and y2_c > y1_c:
                final_geom_data[i] = {
                    "bbox": (x1_c, y1_c, x2_c, y2_c),
                    "kps": data["mesh_pts"] if data and data["type"] == "mp" else (data["kps"] if data else None),
                    "mask": data["seg_mask"] if data and data["type"] == "mp" else None,
                    "is_real": is_real_list[i]
                }
            else:
                final_geom_data[i] = None

        # 第四阶段：正式生成输出 (已在第三阶段完成平滑和插值)
        last_valid_geom = None
        # 预先寻找第一个有效 geom 作为初始兜底
        for g in final_geom_data:
            if g is not None:
                last_valid_geom = g
                break

        last_real_mask = None
        frames_since_last_real = 0
        for i in range(B):
            img = images_np[i]
            geom = final_geom_data[i]
            
            if geom is None:
                if last_real_mask is not None:
                    frames_since_last_real += 1
                hold_active = (hold_max_frame == -1) or (frames_since_last_real <= hold_max_frame)
                
                if last_valid_geom is not None:
                    # 使用最近（或第一个）有效的裁剪区域，避免黑帧
                    x1, y1, x2, y2 = last_valid_geom["bbox"]
                    face_img = img[y1:y2, x1:x2]
                    face_img = cv2.resize(face_img, (output_width, output_heigh))
                    face_images.append(face_img)
                    face_masks.append(np.zeros((output_heigh, output_width), dtype=np.float32))
                    
                    if hold_active:
                        face_masks_hold.append(last_real_mask) # 记录最近的蒙版
                        face_masks_hold_all.append(np.ones((output_heigh, output_width), dtype=np.float32))
                    else:
                        face_masks_hold.append(np.zeros((output_heigh, output_width), dtype=np.float32))
                        face_masks_hold_all.append(np.zeros((output_heigh, output_width), dtype=np.float32))
                        
                    face_bboxes_out.append((x1, y1, x2 - x1, y2 - y1))
                    comfy_pbar.update(1)
                    continue
                else:
                    # 极端兜底：全黑
                    face_images.append(np.zeros((output_heigh, output_width, 3), dtype=np.uint8))
                    face_masks.append(np.zeros((output_heigh, output_width), dtype=np.float32))
                    
                    if hold_active:
                        face_masks_hold.append(None)
                        face_masks_hold_all.append(np.ones((output_heigh, output_width), dtype=np.float32))
                    else:
                        face_masks_hold.append(np.zeros((output_heigh, output_width), dtype=np.float32))
                        face_masks_hold_all.append(np.zeros((output_heigh, output_width), dtype=np.float32))
                        
                    face_bboxes_out.append((0, 0, W, H))
                    comfy_pbar.update(1)
                    continue

            # 更新最近一次有效的 geom
            last_valid_geom = geom
            x1, y1, x2, y2 = geom["bbox"]
            current_face_kps = geom["kps"]
            current_face_mask = geom["mask"]

            # 裁剪图像
            face_img = img[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, (output_width, output_heigh))
            face_images.append(face_img)
            
            # 生成 Mask
            is_real_face = geom.get("is_real", True)
            if not is_real_face:
                # 如果是插值得到的帧（原始帧没有检测到人脸），则输出全黑 Mask
                mask = np.zeros((output_heigh, output_width), dtype=np.float32)
            elif current_face_mask is not None:
                mask_crop = current_face_mask[y1:y2, x1:x2].copy()
                
                # 优化 1：二值化处理，解决半透明问题
                mask_crop = (mask_crop > mask_threshold).astype(np.float32)
                
                # 优化 2：使用 Face Oval 限制范围，防止衣服、背景等区域被误检测为皮肤
                if current_face_kps is not None and len(current_face_kps) > 400:
                    oval_mask = np.zeros_like(mask_crop)
                    # 提取人脸外轮廓点并转换到裁剪空间
                    local_oval_kps = current_face_kps[FACE_OVAL_INDICES].copy()
                    local_oval_kps[:, 0] -= x1
                    local_oval_kps[:, 1] -= y1
                    # 填充人脸轮廓
                    hull = cv2.convexHull(local_oval_kps.astype(np.int32))
                    cv2.fillConvexPoly(oval_mask, hull, 1.0)
                    # 与分割掩码取交集
                    mask_crop = mask_crop * oval_mask
                
                mask = cv2.resize(mask_crop, (output_width, output_heigh), interpolation=cv2.INTER_LINEAR)
                mask = (mask > mask_threshold).astype(np.float32) # Resize 后再次二值化，确保边缘绝对清晰
            else:
                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
                local_kps = current_face_kps.copy()
                local_kps[:, 0] -= x1
                local_kps[:, 1] -= y1
                if len(local_kps) < 100:
                    left_eye = local_kps[15:21].mean(axis=0) if len(local_kps) > 20 else local_kps[0]
                    right_eye = local_kps[22:28].mean(axis=0) if len(local_kps) > 27 else local_kps[-1]
                    eye_dist = np.linalg.norm(left_eye - right_eye)
                    forehead_pt = (left_eye + right_eye)/2 - np.array([0, eye_dist * 0.8])
                    local_kps = np.vstack([local_kps, forehead_pt])
                hull = cv2.convexHull(local_kps.astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 1.0)
                mask = cv2.resize(mask, (output_width, output_heigh), interpolation=cv2.INTER_LINEAR)
                mask = (mask > mask_threshold).astype(np.float32) # 同样二值化处理，解决半透明问题
            
            if is_real_face:
                last_real_mask = mask
                frames_since_last_real = 0
            else:
                if last_real_mask is not None:
                    frames_since_last_real += 1
                
            hold_active = (hold_max_frame == -1) or (frames_since_last_real <= hold_max_frame)

            face_masks.append(mask)
            
            if is_real_face:
                face_masks_hold.append(mask)
                face_masks_hold_all.append(mask)
            else:
                if hold_active:
                    face_masks_hold.append(last_real_mask)
                    face_masks_hold_all.append(np.ones((output_heigh, output_width), dtype=np.float32))
                else:
                    face_masks_hold.append(np.zeros((output_heigh, output_width), dtype=np.float32))
                    face_masks_hold_all.append(np.zeros((output_heigh, output_width), dtype=np.float32))
                    
            face_bboxes_out.append((x1, y1, x2 - x1, y2 - y1))
            comfy_pbar.update(1)

        # 对 face_masks_hold 进行双向填充
        if hold_backward:
            # 向后填充逻辑 (Backward hold: 从“有人”到“无人”的反向覆盖)
            next_real_mask = None
            frames_to_next_real = 0
            for i in range(B - 1, -1, -1):
                is_real = final_geom_data[i] and final_geom_data[i].get("is_real", False)
                if is_real:
                    next_real_mask = face_masks[i]
                    frames_to_next_real = 0
                else:
                    if next_real_mask is not None:
                        frames_to_next_real += 1
                        hold_active = (hold_max_frame == -1) or (frames_to_next_real <= hold_max_frame)
                        if hold_active:
                            # 仅在当前帧没有有效蒙版时（比如 forward hold 没够到）才进行反向填充
                            is_empty = (face_masks_hold[i] is None) or (isinstance(face_masks_hold[i], np.ndarray) and np.all(face_masks_hold[i] == 0))
                            if is_empty:
                                face_masks_hold[i] = next_real_mask
                                face_masks_hold_all[i] = np.ones((output_heigh, output_width), dtype=np.float32)
        else:
            # 原始逻辑：仅填充序列开头的空缺（不限帧数，直到遇到第一个真实蒙版）
            first_real_mask = None
            for m in face_masks_hold:
                if m is not None and not (isinstance(m, np.ndarray) and np.all(m == 0)):
                    first_real_mask = m
                    break
            
            if first_real_mask is not None:
                for i in range(B):
                    is_empty = (face_masks_hold[i] is None) or (isinstance(face_masks_hold[i], np.ndarray) and np.all(face_masks_hold[i] == 0))
                    if is_empty:
                        face_masks_hold[i] = first_real_mask
                    else:
                        break # 遇到第一个非空，说明进入了检测区或 forward hold 区
        
        # 统一清理可能遗留的 None
        face_masks_hold = [m if m is not None else np.zeros((output_heigh, output_width), dtype=np.float32) for m in face_masks_hold]

        interpolated_count = sum(1 for d in final_geom_data if d is not None and not d.get("is_real", True))
        print(f"[BatchFaceDetection] Summary: MediaPipe(468pts)={mp_count}, YOLO+ViTPose(Fallback)={fallback_count}, Interpolated(No Mask)={interpolated_count}")
        
        # === Generate Pose Video ===
        images_pose_tensor = torch.zeros((B, H, W, 3))
        face_images_pose_tensor = torch.zeros((B, output_heigh, output_width, 3))
        
        if output_pose_video or output_face_pose:
            # 1. Get Reference Landmarks
            ref_landmarks = None
            if ref_image is not None:
                # ref_image is [B, H, W, C], take the first one
                ref_np = ref_image[0].cpu().numpy()
                ref_u8 = (ref_np * 255).clip(0, 255).astype(np.uint8)
                ref_mesh_pts, _, _ = mp_detector.get_face_data(ref_u8)
                if ref_mesh_pts is not None:
                    ref_landmarks = ref_mesh_pts
            
            if ref_landmarks is None:
                # Use first valid detection as reference
                for d in raw_detections:
                    if d is not None:
                        if d["type"] == "mp":
                            ref_landmarks = d["mesh_pts"]
                        else:
                            # If it's yolo, we use the 69 points as reference
                            ref_landmarks = d["kps"]
                        break
            
            if ref_landmarks is not None:
                # 2. Extract video landmarks
                video_landmarks_list = []
                for d in raw_detections:
                    if d is not None:
                        video_landmarks_list.append(d["mesh_pts"] if d["type"] == "mp" else d["kps"])
                    else:
                        video_landmarks_list.append(None)
                
                # 3. Align
                aligner = FaceMeshAlign()
                success, aligned_landmarks = aligner(video_landmarks_list, ref_landmarks)
                
                if success:
                    full_pose_frames = np.zeros((B, H, W, 3), dtype=np.uint8) if output_pose_video else None
                    face_pose_frames = np.zeros((B, output_heigh, output_width, 3), dtype=np.uint8) if output_face_pose else None
                    
                    for frame_idx in range(B):
                        # 如果该帧原本未检测到人脸（是通过插值补齐的），则保持全黑
                        d = raw_detections[frame_idx]
                        geom = final_geom_data[frame_idx]
                        if d is None or not d.get("is_real", True) or geom is None:
                            continue

                        landmarks = aligned_landmarks[frame_idx]
                        num_pts = landmarks.shape[0]
                        
                        # Draw core landmarks as white points
                        if output_pose_video:
                            frame = full_pose_frames[frame_idx]
                            for idx in CORE_LANDMARK_INDICES:
                                if idx < num_pts:
                                    x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
                                    if 0 <= x < W and 0 <= y < H:
                                        cv2.circle(frame, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
                                        
                        # Draw cropped face images pose
                        if output_face_pose:
                            face_frame = face_pose_frames[frame_idx]
                            x1, y1, x2, y2 = geom["bbox"]
                            scale_x = output_width / (x2 - x1)
                            scale_y = output_heigh / (y2 - y1)
                            
                            for idx in CORE_LANDMARK_INDICES:
                                if idx < num_pts:
                                    # 转换到裁剪框坐标并缩放
                                    lx = int((landmarks[idx, 0] - x1) * scale_x)
                                    ly = int((landmarks[idx, 1] - y1) * scale_y)
                                    if 0 <= lx < output_width and 0 <= ly < output_heigh:
                                        cv2.circle(face_frame, (lx, ly), radius=2, color=(255, 255, 255), thickness=-1)
                        
                        comfy_pbar.update(1)
                    
                    if output_pose_video:
                        images_pose_tensor = torch.from_numpy(full_pose_frames.astype(np.float32) / 255.0)
                    if output_face_pose:
                        face_images_pose_tensor = torch.from_numpy(face_pose_frames.astype(np.float32) / 255.0)

        detector.cleanup()
        pose_model.cleanup()
        mp_detector.close()

        face_images_tensor = torch.from_numpy(np.stack(face_images, 0))
        face_masks_tensor = torch.from_numpy(np.stack(face_masks, 0))
        face_masks_hold_tensor = torch.from_numpy(np.stack(face_masks_hold, 0))
        face_masks_hold_all_tensor = torch.from_numpy(np.stack(face_masks_hold_all, 0))

        return (face_images_tensor, face_masks_tensor, face_masks_hold_tensor, face_masks_hold_all_tensor, face_bboxes_out, images_pose_tensor, face_images_pose_tensor)
    
def to_numpy(img):
    if img is None:
        return None
    if isinstance(img, list):
        img = img[0]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.array(img)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

class FaceMergeNode:
    """执行单帧或批量人脸贴回的核心逻辑"""
    _mask_cache = {}  # 类级别缓存，跨实例共享

    def __init__(self, blend_mode="poisson", feather_amount=0.1, resize_fit=True):
        self.blend_mode = blend_mode
        self.feather_amount = feather_amount
        self.resize_fit = resize_fit

    def get_feather_mask(self, h, w):
        key = (h, w, self.feather_amount)
        if key in self._mask_cache:
            return self._mask_cache[key]
        
        mask = np.ones((h, w), dtype=np.uint8) * 255
        feather_size = self.feather_amount * max(w, h)
        if feather_size > 0:
            mask_f = cv2.GaussianBlur(mask, (0, 0), feather_size)
        else:
            mask_f = mask
        mask_f = (mask_f / 255.0)[..., None].astype(np.float32)
        
        # 限制缓存大小，防止内存溢出
        if len(self._mask_cache) > 100:
            self._mask_cache.clear()
        self._mask_cache[key] = mask_f
        return mask_f

    def process_single_face(self, ori, face, bbox, mask=None):
        """处理单张图像与对应bbox"""
        x, y, w, h = map(int, bbox)

        # 确保输入是 numpy 数组且格式正确
        if not isinstance(ori, np.ndarray):
            ori = to_numpy(ori)
        if not isinstance(face, np.ndarray):
            face = to_numpy(face)

        if self.resize_fit:
            if face.shape[0] != h or face.shape[1] != w:
                face = cv2.resize(face, (w, h))

        result = ori.copy()

        # 🔹 混合模式
        if self.blend_mode == "poisson":
            try:
                # 创建白色mask
                p_mask = np.ones((h, w), dtype=np.uint8) * 255
                center = (x + w // 2, y + h // 2)
                result = cv2.seamlessClone(face, ori, p_mask, center, cv2.NORMAL_CLONE)
            except Exception as e:
                # print(f"[FaceMergeNode] Poisson blending failed: {e}")
                result[y:y+h, x:x+w] = face

        elif self.blend_mode == "feather":
            mask_f = self.get_feather_mask(h, w)
            roi = ori[y:y+h, x:x+w].astype(np.float32)
            face_f = face.astype(np.float32)
            blended = face_f * mask_f + roi * (1.0 - mask_f)
            result[y:y+h, x:x+w] = np.clip(blended, 0, 255).astype(np.uint8)

        elif self.blend_mode == "mask":
            if mask is not None:
                # 处理 mask 尺寸和通道
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h))
                if len(mask.shape) == 2:
                    mask_f = mask[..., None].astype(np.float32)
                else:
                    mask_f = mask.astype(np.float32)
                
                # 确保 mask 在 [0, 1] 范围
                if mask_f.max() > 1.0:
                    mask_f /= 255.0
                
                roi = ori[y:y+h, x:x+w].astype(np.float32)
                face_f = face.astype(np.float32)
                blended = face_f * mask_f + roi * (1.0 - mask_f)
                result[y:y+h, x:x+w] = np.clip(blended, 0, 255).astype(np.uint8)
            else:
                # 如果没有提供 mask，退化为直接贴图
                result[y:y+h, x:x+w] = face

        elif self.blend_mode == "alpha":
            # alpha 模式在此上下文中等同于直接覆盖 (direct)
            result[y:y+h, x:x+w] = face

        else:
            result[y:y+h, x:x+w] = face

        return result

    def process_image(self, original_images, edited_faces, face_bboxes, masks=None):
        """批量处理多张图像"""
        # 预先转换整个批次，避免在循环中重复转换和 CPU/GPU 传输
        def prepare_images(imgs):
            if isinstance(imgs, torch.Tensor):
                # 假设 ComfyUI 标准的 [0, 1] 范围张量
                return (imgs.detach().cpu().numpy() * 255).astype(np.uint8)
            elif isinstance(imgs, list):
                return [to_numpy(x) for x in tqdm(imgs, desc="Preparing Images", leave=False)]
            return imgs

        def prepare_masks(m):
            if m is None:
                return None
            if isinstance(m, torch.Tensor):
                # ComfyUI Mask 是 (B, H, W)，需要转换为 numpy
                return m.detach().cpu().numpy()
            return m

        ori_np = prepare_images(original_images)
        face_np = prepare_images(edited_faces)
        masks_np = prepare_masks(masks)
        
        num = len(ori_np)
        results = []
        
        pbar = ProgressBar(num)
        for i in tqdm(range(num), desc="Merging Faces"):
            ori = ori_np[i]
            face = face_np[i] if len(face_np) > i else face_np[0]
            bbox = face_bboxes[i] if len(face_bboxes) > i else face_bboxes[0]
            mask = masks_np[i] if masks_np is not None and len(masks_np) > i else (masks_np[0] if masks_np is not None else None)
            
            result = self.process_single_face(ori, face, bbox, mask=mask)
            results.append(result)
            pbar.update(1)

        return np.stack(results, 0)


class BatchFaceRestoreToSource:
    """ComfyUI 节点定义"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "edited_faces": ("IMAGE",),
                "face_bboxs": ("BBOX",),
                "blend_mode": (["poisson", "feather", "mask", "alpha", "direct"],),
                "feather_amount": ("FLOAT", {"default": 0.1, "min": 0, "max": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "BatchFaceDetection"

    def merge(self, original_images, edited_faces, face_bboxs, blend_mode, feather_amount, mask=None):
        node = FaceMergeNode(blend_mode, feather_amount)
        out = node.process_image(original_images, edited_faces, face_bboxs, masks=mask)
        return (torch.from_numpy(out).float() / 255.0,)


# ==================== 节点映射 ====================

NODE_CLASS_MAPPINGS = {
    "BatchFaceDetectionModelLoader": BatchFaceDetectionModelLoader,
    "BatchFaceDetection": BatchFaceDetection,
    "BatchFaceRestoreToSource": BatchFaceRestoreToSource
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchFaceDetectionModelLoader": "Batch Face Detection Model Loader",
    "BatchFaceDetection": "Batch Face Detection",
    "BatchFaceRestoreToSource": "Batch Face Restore To Source"
}
