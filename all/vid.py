import cv2
import numpy as np
import onnxruntime as ort
import time

# ===== CONFIGURATION =====
MODEL_PATH = "best3.onnx"
CLASSES_PATH = "classes.txt"
VIDEO_SOURCE = "1230.mp4"  # Use 0 for Webcam
IMG_SIZE = 640
CONF_THRESH = 0.25
NMS_THRESH = 0.45

# ===== INITIALIZE MODEL =====
# Raspberry Pi 5 uses CPUExecutionProvider by default
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
CLASSES = open(CLASSES_PATH, "r", encoding="utf-8").read().strip().split("\n")

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image to 640x640 while maintaining aspect ratio using padding"""
    shape = img.shape[:2] # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, left, top

def preprocess(img):
    img_lb, r, pad_w, pad_h = letterbox(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = img_norm.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_chw, 0)
    return img_batch, r, pad_w, pad_h

def postprocess(preds, frame, r, pad_w, pad_h):
    # YOLOv8 output: [1, 4 + num_classes, 8400]
    preds = np.squeeze(preds).T # Transpose to [8400, 4 + num_classes]
    
    boxes_xywh = preds[:, :4]
    scores = np.max(preds[:, 4:], axis=1)
    class_ids = np.argmax(preds[:, 4:], axis=1)

    # Filter by confidence threshold
    idx = scores > CONF_THRESH
    boxes_xywh, scores, class_ids = boxes_xywh[idx], scores[idx], class_ids[idx]

    if len(boxes_xywh) == 0: 
        return frame, 0, 0

    # Rescale coordinates to original image dimensions
    x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_w) / r
    y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_h) / r
    x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - pad_w) / r
    y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pad_h) / r
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS (Non-Maximum Suppression)
    boxes_wh = boxes_xyxy.copy()
    boxes_wh[:, 2:] -= boxes_wh[:, :2] # xyxy to xywh
    indices = cv2.dnn.NMSBoxes(boxes_wh.tolist(), scores.tolist(), CONF_THRESH, NMS_THRESH)

    count_empty = 0
    count_occupied = 0
    COLORS = [(0, 255, 0), (0, 0, 255)] # Green for Empty, Red for Occupied

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes_xyxy[i].astype(int)
            cls_id = class_ids[i]
            score = scores[i]
            
            if cls_id == 0: count_empty += 1
            else: count_occupied += 1

            # Draw bounding box and label
            color = COLORS[cls_id] if cls_id < len(COLORS) else (255, 0, 0)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{CLASSES[cls_id]} {score:.2f}"
            cv2.putText(frame, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame, count_empty, count_occupied

# ===== MAIN EXECUTION =====
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Processing started. Press 'q' to quit.")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret: 
        break

    # 1. Preprocessing
    blob, r, pad_w, pad_h = preprocess(frame)

    # 2. Model Inference
    outputs = session.run(None, {input_name: blob})

    # 3. Postprocessing & Drawing
    result_frame, empty, occupied = postprocess(outputs[0], frame, r, pad_w, pad_h)

    # 4. Calculate FPS and display stats
    fps = 1 / (time.time() - start_time)
    info_text = f"FPS: {fps:.1f} | Empty: {empty} | Occupied: {occupied}"
    cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 5. Show Frame
    cv2.imshow("YOLOv8 PKLot - Raspberry Pi 5", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Inference finished.")


