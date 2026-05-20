import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model.MLT import MLT
from Pytorch_Retinaface.data import cfg_mnet
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from STAR.demo import Alignment, draw_pts


EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


def detect_faces(image, retinaface_model, device, resize=1, confidence_threshold=0.02, nms_threshold=0.4, vis_threshold=0.5):
    img = np.float32(image)
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
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet["variance"])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet["variance"])
    scale1 = torch.Tensor([
        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        img.shape[3], img.shape[2],
    ]).to(device)
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
    return dets[dets[:, 4] >= vis_threshold]


def detection_keypoints(det):
    return det[5:15].reshape(5, 2)


def clamp_bbox(bbox, image_shape):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    return x1, y1, x2, y2


def validate_retinaface_geometry(det, image_shape, min_confidence=0.8):
    if float(det[4]) < min_confidence:
        return False, f"low face confidence {float(det[4]):.4f} < {min_confidence:.2f}"

    x1, y1, x2, y2 = clamp_bbox(det, image_shape)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return False, "invalid bbox"

    aspect_ratio = width / height
    if not 0.45 <= aspect_ratio <= 1.45:
        return False, f"unusual bbox aspect ratio {aspect_ratio:.2f}"

    keypoints = detection_keypoints(det)
    if not np.isfinite(keypoints).all():
        return False, "non-finite RetinaFace keypoints"

    margin_x = width * 0.08
    margin_y = height * 0.08
    inside_x = (keypoints[:, 0] >= x1 - margin_x) & (keypoints[:, 0] <= x2 + margin_x)
    inside_y = (keypoints[:, 1] >= y1 - margin_y) & (keypoints[:, 1] <= y2 + margin_y)
    if not np.all(inside_x & inside_y):
        return False, "RetinaFace keypoints fall outside bbox"

    left_eye, right_eye, nose, left_mouth, right_mouth = keypoints
    eye_center = (left_eye + right_eye) / 2
    mouth_center = (left_mouth + right_mouth) / 2
    eye_distance = np.linalg.norm(left_eye - right_eye)
    mouth_distance = np.linalg.norm(left_mouth - right_mouth)

    if not 0.15 * width <= eye_distance <= 0.75 * width:
        return False, f"eye distance ratio {eye_distance / width:.2f} out of range"
    if not 0.10 * width <= mouth_distance <= 0.80 * width:
        return False, f"mouth distance ratio {mouth_distance / width:.2f} out of range"
    if abs(left_eye[1] - right_eye[1]) > 0.25 * height:
        return False, "eyes are not horizontally consistent"
    if abs(left_mouth[1] - right_mouth[1]) > 0.25 * height:
        return False, "mouth corners are not horizontally consistent"
    if mouth_center[1] <= eye_center[1] + 0.12 * height:
        return False, "mouth is not below eyes"
    if not eye_center[1] - 0.05 * height <= nose[1] <= mouth_center[1] + 0.10 * height:
        return False, "nose is not between eyes and mouth"

    min_eye_x = min(left_eye[0], right_eye[0])
    max_eye_x = max(left_eye[0], right_eye[0])
    if not min_eye_x - 0.20 * width <= nose[0] <= max_eye_x + 0.20 * width:
        return False, "nose is not centered near the eyes"

    return True, "ok"


def validate_star_landmarks(landmarks, bbox, image_shape):
    if landmarks is None or len(landmarks) < 88:
        return False, "STAR landmarks missing or incomplete"
    if not np.isfinite(landmarks).all():
        return False, "non-finite STAR landmarks"

    x1, y1, x2, y2 = clamp_bbox(bbox, image_shape)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return False, "invalid bbox"

    margin_x = width * 0.12
    margin_y = height * 0.12
    inside_x = (landmarks[:, 0] >= x1 - margin_x) & (landmarks[:, 0] <= x2 + margin_x)
    inside_y = (landmarks[:, 1] >= y1 - margin_y) & (landmarks[:, 1] <= y2 + margin_y)
    inside_ratio = float(np.mean(inside_x & inside_y))
    if inside_ratio < 0.72:
        return False, f"STAR landmark inside-bbox ratio {inside_ratio:.2f} < 0.72"

    spread_x = float(np.max(landmarks[:, 0]) - np.min(landmarks[:, 0]))
    spread_y = float(np.max(landmarks[:, 1]) - np.min(landmarks[:, 1]))
    if not 0.30 * width <= spread_x <= 1.15 * width:
        return False, f"STAR landmark width ratio {spread_x / width:.2f} out of range"
    if not 0.30 * height <= spread_y <= 1.15 * height:
        return False, f"STAR landmark height ratio {spread_y / height:.2f} out of range"

    left_eye = landmarks[60:68].mean(axis=0)
    right_eye = landmarks[68:76].mean(axis=0)
    mouth = landmarks[76:88]
    eye_center = (left_eye + right_eye) / 2
    mouth_center = mouth.mean(axis=0)
    eye_distance = float(np.linalg.norm(left_eye - right_eye))
    mouth_width = float(np.max(mouth[:, 0]) - np.min(mouth[:, 0]))

    if not 0.15 * width <= eye_distance <= 0.75 * width:
        return False, f"STAR eye distance ratio {eye_distance / width:.2f} out of range"
    if abs(left_eye[1] - right_eye[1]) > 0.20 * height:
        return False, "STAR eyes are not horizontally consistent"
    if mouth_center[1] <= eye_center[1] + 0.12 * height:
        return False, "STAR mouth is not below eyes"
    if not 0.10 * width <= mouth_width <= 0.75 * width:
        return False, f"STAR mouth width ratio {mouth_width / width:.2f} out of range"

    return True, "ok"


def face_tensor_from_bbox(image, bbox, device):
    x1, y1, x2, y2 = clamp_bbox(bbox, image.shape)
    if x2 <= x1 or y2 <= y1:
        return None, None

    cropped_face = image[y1:y2, x1:x2]
    rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(rgb_face)
    face_tensor = transform(pil_face).unsqueeze(0).to(device)
    return face_tensor, cropped_face


def analyze_landmarks(image, bbox, alignment):
    x1, y1, x2, y2 = clamp_bbox(bbox, image.shape)
    if x2 <= x1 or y2 <= y1:
        return None

    center_w = (x2 + x1) / 2
    center_h = (y2 + y1) / 2
    scale = min(x2 - x1, y2 - y1) / 200 * 1.05
    return alignment.analyze(image, float(scale), float(center_w), float(center_h))


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
    if landmarks is None or len(landmarks) == 0:
        return []

    if len(landmarks) > 97:
        return [(int(landmarks[96, 0]), int(landmarks[96, 1])), (int(landmarks[97, 0]), int(landmarks[97, 1]))]
    if len(landmarks) >= 68:
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        return [(int(left_eye[0]), int(left_eye[1])), (int(right_eye[0]), int(right_eye[1]))]
    return []


def draw_gaze_arrows(image, gaze_output, landmarks, bbox):
    gaze_vector = gaze_to_3d(gaze_output[0])
    x1, y1, x2, y2 = clamp_bbox(bbox, image.shape)
    face_size = max(1, min(x2 - x1, y2 - y1))
    scale = max(70, int(face_size * 0.9))
    dx = int(gaze_vector[0] * scale)
    dy = int(gaze_vector[1] * scale)

    centers = eye_centers_from_landmarks(landmarks)
    if not centers:
        centers = [((x1 + x2) // 2, (y1 + y2) // 2)]

    for center in centers:
        end_point = (center[0] + dx, center[1] + dy)
        cv2.arrowedLine(image, center, end_point, (0, 255, 255), 2, tipLength=0.25)
        cv2.circle(image, center, 3, (0, 180, 255), -1)


def draw_label(image, bbox, face_index, probabilities, best_index):
    x1, y1, x2, y2 = clamp_bbox(bbox, image.shape)
    label = f"Face {face_index}: {EMOTION_LABELS[best_index]} {probabilities[best_index]:.1f}%"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    label_y = y1 - 8 if y1 - text_height - 10 > 0 else y1 + text_height + 10
    bg_top = max(0, label_y - text_height - baseline - 4)
    bg_bottom = min(image.shape[0], label_y + baseline + 4)
    bg_right = min(image.shape[1], x1 + text_width + 8)
    cv2.rectangle(image, (x1, bg_top), (bg_right, bg_bottom), (0, 200, 0), -1)
    cv2.putText(image, label, (x1 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def compact_emotion_items(probabilities):
    return [f"{label} {probability:.1f}%" for label, probability in zip(EMOTION_LABELS, probabilities)]


def add_multiface_panel(image, summaries):
    if not summaries:
        return image

    lines = []
    for summary in summaries:
        lines.append(
            f"Face {summary['face_index']} Top: "
            f"{EMOTION_LABELS[summary['best_index']]} {summary['probabilities'][summary['best_index']]:.1f}%"
        )
        items = compact_emotion_items(summary["probabilities"])
        if image.shape[1] >= 760:
            lines.append(" | ".join(items))
        elif image.shape[1] >= 420:
            lines.extend([" | ".join(items[:4]), " | ".join(items[4:])])
        else:
            lines.extend([" | ".join(items[index:index + 2]) for index in range(0, len(items), 2)])

    line_height = 20
    panel_height = 20 + line_height * len(lines)
    panel = np.full((panel_height, image.shape[1], 3), 255, dtype=np.uint8)
    y = 24
    for line in lines:
        color = (0, 100, 0) if " Top: " in line else (0, 0, 0)
        thickness = 2 if " Top: " in line else 1
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, thickness, cv2.LINE_AA)
        y += line_height
    return np.vstack((image, panel))


def save_cropped_face(cropped_face, cropped_output_dir, face_index):
    if not cropped_output_dir or cropped_face is None:
        return None

    os.makedirs(cropped_output_dir, exist_ok=True)
    cropped_path = os.path.join(cropped_output_dir, f"face_{face_index}.jpg")
    cv2.imwrite(cropped_path, cropped_face)
    return cropped_path


def run_multiface_demo(model, retinaface_model, alignment, image_path, device, output_path, cropped_output_dir, args):
    image_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_raw is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    dets = detect_faces(image_raw, retinaface_model, device, vis_threshold=args.face_threshold)
    if len(dets) == 0:
        print("No face detected.")
        return

    model.eval()
    image_draw = image_raw.copy()
    summaries = []

    for candidate_index, det in enumerate(dets):
        if not args.disable_face_filter:
            is_valid, reason = validate_retinaface_geometry(det, image_raw.shape, args.filter_min_confidence)
            if not is_valid:
                print(f"Candidate {candidate_index} skipped by RetinaFace geometry filter: {reason}")
                continue

        landmarks = analyze_landmarks(image_raw, det, alignment)
        if not args.disable_face_filter:
            is_valid, reason = validate_star_landmarks(landmarks, det, image_raw.shape)
            if not is_valid:
                print(f"Candidate {candidate_index} skipped by STAR landmark filter: {reason}")
                continue

        face_tensor, cropped_face = face_tensor_from_bbox(image_raw, det, device)
        if face_tensor is None:
            print(f"Candidate {candidate_index}: invalid bounding box, skipped.")
            continue

        face_index = len(summaries)
        cropped_path = save_cropped_face(cropped_face, cropped_output_dir, face_index)
        with torch.no_grad():
            emotion_output, gaze_output, au_output = model(face_tensor)

        if landmarks is not None:
            image_draw = draw_pts(image_draw, landmarks)

        probabilities, best_index = emotion_summary(emotion_output)
        draw_gaze_arrows(image_draw, gaze_output, landmarks, det)
        draw_label(image_draw, det, face_index, probabilities, best_index)

        summary = {
            "face_index": face_index,
            "candidate_index": candidate_index,
            "bbox": [round(float(value), 2) for value in det[:4]],
            "confidence": round(float(det[4]), 4),
            "probabilities": probabilities,
            "best_index": best_index,
            "cropped_path": cropped_path,
        }
        summaries.append(summary)

        emotion_percentages = {
            label: round(float(probability), 1)
            for label, probability in zip(EMOTION_LABELS, probabilities)
        }
        print(f"Face {face_index}: candidate={candidate_index} bbox={summary['bbox']} confidence={summary['confidence']}")
        print(f"Face {face_index} Top Emotion: {EMOTION_LABELS[best_index]} ({probabilities[best_index]:.1f}%)")
        print(f"Face {face_index} Emotion Percentages: {emotion_percentages}")
        print(f"Face {face_index} Gaze Output: {gaze_output}")
        print(f"Face {face_index} AU Output: {au_output}")
        if cropped_path:
            print(f"Face {face_index} cropped image: {cropped_path}")

    if not summaries:
        print("No valid face crops were processed.")
        return

    image_draw = add_multiface_panel(image_draw, summaries)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, image_draw)
    print(f"Saved multiface visualization: {output_path}")
    print(f"Processed faces: {len(summaries)}")


def build_models(args, device):
    model = MLT()
    model.load_state_dict(torch.load(args.mtl_model, map_location=device))
    model = model.to(device)
    model.eval()

    retinaface_model = RetinaFace(cfg=cfg_mnet, phase="test")
    retinaface_model = load_retinaface_model(retinaface_model, args.retinaface_model, device)
    retinaface_model = retinaface_model.to(device)
    retinaface_model.eval()

    device_id = 0 if device.type == "cuda" and device.index is None else device.index
    alignment_args = argparse.Namespace(
        config_name="alignment",
        device_id=device_id if device.type == "cuda" else -1,
    )
    device_ids = [device_id] if device.type == "cuda" else [-1]
    alignment = Alignment(alignment_args, args.landmark_model, dl_framework="pytorch", device_ids=device_ids)

    return model, retinaface_model, alignment


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenFace-3.0 multiface image demo.")
    parser.add_argument("--image", default="images/89.jpg", help="Input image path.")
    parser.add_argument("--output", default="images/test_out_multiface.png", help="Output visualization path.")
    parser.add_argument("--cropped-output-dir", default="images/cropped_faces", help="Directory for per-face cropped images.")
    parser.add_argument("--mtl-model", default="./weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth", help="Multitask model weight path.")
    parser.add_argument("--retinaface-model", default="./weights/mobilenet0.25_Final.pth", help="RetinaFace model weight path.")
    parser.add_argument("--landmark-model", default="./weights/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl", help="STAR landmark model weight path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--face-threshold", type=float, default=0.5, help="Initial RetinaFace detection score threshold.")
    parser.add_argument("--filter-min-confidence", type=float, default=0.8, help="Minimum RetinaFace score for the geometry post-filter.")
    parser.add_argument("--disable-face-filter", action="store_true", help="Disable RetinaFace keypoint and STAR landmark geometry post-filtering.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, retinaface_model, alignment = build_models(args, device)
    run_multiface_demo(
        model,
        retinaface_model,
        alignment,
        args.image,
        device,
        args.output,
        args.cropped_output_dir,
        args,
    )
