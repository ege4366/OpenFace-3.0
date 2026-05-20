import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.multiprocessing as mp
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import os
import argparse

import torch.nn.functional as F
import torch.optim as optim
import cv2

from model.MLT import MLT

from model.AutomaticWeightedLoss import AutomaticWeightedLoss
from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from Pytorch_Retinaface.data import cfg_mnet, cfg_re50

from STAR.demo import GetCropMatrix, TransformPerspective, TransformPoints2D, Alignment, draw_pts


EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]


def load_retinaface_model(model, pretrained_path, device):
    print(f"Loading pretrained model from {pretrained_path}")
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    if "state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["state_dict"]
    pretrained_dict = {
        key.removeprefix("module."): value
        for key, value in pretrained_dict.items()
    }
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def preprocess_image(image_path, retinaface_model, device, resize=1, confidence_threshold=0.02, nms_threshold=0.4, vis_thres=0.5):
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_raw is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = np.float32(img_raw)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(device)  
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)  

    with torch.no_grad():
        loc, conf, landms = retinaface_model(img)

    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)  
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]]).to(device)  
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    dets = np.concatenate((dets, landms), axis=1)

    print(dets)
    
    if len(dets) == 0:
        return None, None

    conf = dets[0][4]
    b = dets[0].astype(int) 
    print(b)
    if conf < vis_thres:
        return None, None

    face = img_raw[b[1]:b[3], b[0]:b[2]]
    face = Image.fromarray(face)
    face = transform(face).unsqueeze(0).to(device)  


    return face, dets


def landmark_detection(image, dets, alignment):
    results = []
    for det in dets:
        x1, y1, x2, y2 = det[:4].astype(int) 
        conf = det[4]
        print(x1, y1, x2, y2, conf )
        if conf < 0.5:  
            continue
        
        face = image[y1:y2, x1:x2]
        center_w = (x2 + x1) / 2
        center_h = (y2 + y1) / 2
        scale = min(x2 - x1, y2 - y1) / 200 * 1.05
        
        landmarks_pv = alignment.analyze(image, float(scale), float(center_w), float(center_h))
        results.append(landmarks_pv)
        image = draw_pts(image, landmarks_pv)
    return image, results


def emotion_summary(emotion_output):
    probabilities = torch.softmax(emotion_output[0], dim=0).detach().cpu().numpy() * 100
    best_index = int(np.argmax(probabilities))
    return probabilities, best_index


def gaze_to_3d(gaze_output):
    yaw, pitch = gaze_output.detach().cpu().numpy().tolist()
    return np.array([
        -np.cos(pitch) * np.sin(yaw),
        -np.sin(pitch),
        -np.cos(pitch) * np.cos(yaw),
    ], dtype=np.float32)


def eye_centers_from_landmarks(landmarks):
    if not landmarks:
        return []

    points = landmarks[0]
    if points is None or len(points) == 0:
        return []

    if len(points) > 97:
        return [(int(points[96, 0]), int(points[96, 1])), (int(points[97, 0]), int(points[97, 1]))]
    if len(points) >= 68:
        left_eye = points[36:42].mean(axis=0)
        right_eye = points[42:48].mean(axis=0)
        return [(int(left_eye[0]), int(left_eye[1])), (int(right_eye[0]), int(right_eye[1]))]
    return []


def draw_gaze_arrows(image, gaze_output, landmarks):
    gaze_vector = gaze_to_3d(gaze_output[0])
    scale = max(120, int(min(image.shape[:2]) * 1.5))
    dx = int(gaze_vector[0] * scale)
    dy = int(gaze_vector[1] * scale)

    image_with_gaze = image.copy()
    for center in eye_centers_from_landmarks(landmarks):
        end_point = (center[0] + dx, center[1] + dy)
        cv2.arrowedLine(image_with_gaze, center, end_point, (0, 255, 255), 2, tipLength=0.25)
        cv2.circle(image_with_gaze, center, 3, (0, 180, 255), -1)
    return image_with_gaze


def add_emotion_panel(image, probabilities, best_index):
    use_two_columns = image.shape[1] >= 420
    rows = 4 if use_two_columns else 8
    panel_height = 54 + rows * 22
    panel = np.full((panel_height, image.shape[1], 3), 255, dtype=np.uint8)

    top_text = f"Top: {EMOTION_LABELS[best_index]} {probabilities[best_index]:.1f}%"
    cv2.putText(panel, top_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 110, 0), 2, cv2.LINE_AA)

    col_width = max(1, image.shape[1] // 2) if use_two_columns else image.shape[1]
    for index, (label, probability) in enumerate(zip(EMOTION_LABELS, probabilities)):
        col = index % 2 if use_two_columns else 0
        row = index // 2 if use_two_columns else index
        x = 10 + col * col_width
        y = 54 + row * 20
        text = f"{label}: {probability:.1f}%"
        color = (0, 0, 0) if index != best_index else (0, 120, 0)
        thickness = 1 if index != best_index else 2
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, thickness, cv2.LINE_AA)

    return np.vstack((image, panel))


def demo(model, retinaface_model, alignment, image_path, device, output_path, cropped_output_path):
    image_raw = cv2.imread(image_path)
    face, dets = preprocess_image(image_path, retinaface_model, device)

    model.eval()

    if dets is None or len(dets) == 0:
        print("No face detected.")
        return

    x1, y1, x2, y2= dets[0][:4]

    # Crop the face using array slicing
    cropped_face = image_raw[int(y1):int(y2), int(x1):int(x2)]
    if cropped_output_path:
        cropped_dir = os.path.dirname(cropped_output_path)
        if cropped_dir:
            os.makedirs(cropped_dir, exist_ok=True)
        cv2.imwrite(cropped_output_path, cropped_face)



    pil_image = Image.open(image_path).convert("RGB")
    image = transform(pil_image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        emotion_output, gaze_output, au_output = model(image)

    image_draw, landmarks = landmark_detection(image_raw, dets, alignment)
    if image_draw is None or landmarks is None:
        print("No landmarks detected.")
        image_draw = image_raw.copy()
        landmarks = []

    probabilities, best_index = emotion_summary(emotion_output)
    image_draw = draw_gaze_arrows(image_draw, gaze_output, landmarks)
    image_draw = add_emotion_panel(image_draw, probabilities, best_index)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, image_draw)
    print(f"Saved annotated visualization: {output_path}")
    print(f"Top Emotion: {EMOTION_LABELS[best_index]} ({probabilities[best_index]:.1f}%)")
    print("Emotion Percentages:", {label: round(float(prob), 1) for label, prob in zip(EMOTION_LABELS, probabilities)})
    print("Emotion Output:", emotion_output)
    print("Gaze Output:", gaze_output)
    print("AU Output:", au_output)



transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


import time

def measure_inference_time(model, input_tensor, device, num_runs=200):
    # Move model to the correct device
    model.to(device)
    model.eval()

    # Warm-up to avoid any setup overhead in timing
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Measure the time for multiple runs to get an average
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.time()

    # Calculate average time per run
    avg_time_per_run = (end_time - start_time) / num_runs
    return avg_time_per_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run OpenFace-3.0 image demo.")
    parser.add_argument("--image", default="images/89.jpg", help="Input image path.")
    parser.add_argument("--output", default="images/test_out.png", help="Output visualization path.")
    parser.add_argument("--cropped-output", default="images/cropped_face.jpg", help="Output cropped face path.")
    parser.add_argument("--mtl-model", default="./weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth", help="Multitask model weight path.")
    parser.add_argument("--retinaface-model", default="./weights/mobilenet0.25_Final.pth", help="RetinaFace model weight path.")
    parser.add_argument("--landmark-model", default="./weights/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl", help="STAR landmark model weight path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = MLT()  
    model.load_state_dict(torch.load(args.mtl_model, map_location=device))
    model.eval()

    model = model.to(device)

    cfg = cfg_mnet 
    retinaface_model = RetinaFace(cfg=cfg, phase='test')
    retinaface_model = load_retinaface_model(retinaface_model, args.retinaface_model, device)
    retinaface_model.eval()
    retinaface_model = retinaface_model.to(device)

    device_id = 0 if device.type == 'cuda' and device.index is None else device.index
    config = {
        "config_name": 'alignment',
        "device_id": device_id if device.type == 'cuda' else -1,
    }
    alignment_args = argparse.Namespace(**config)
    device_ids = [device_id] if device.type == 'cuda' else [-1]
    alignment = Alignment(alignment_args, args.landmark_model, dl_framework="pytorch", device_ids=device_ids)

    demo(model, retinaface_model, alignment, args.image, device, args.output, args.cropped_output)
