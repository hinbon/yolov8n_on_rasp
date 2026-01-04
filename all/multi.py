import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
import queue

# ===== CONFIGURATION =====
MODEL_PATH = "best3.onnx"
CLASSES_PATH = "classes.txt"
VIDEO_SOURCE = "1230.mp4" 
IMG_SIZE = 640
# Increase CONF_THRESH to 0.4 or 0.5 to reduce "noise" (false detections)
CONF_THRESH = 0.45 
# Decrease NMS_THRESH if you still see overlapping boxes (e.g., 0.35)
NMS_THRESH = 0.40  

# Load classes
CLASSES = open(CLASSES_PATH, "r", encoding="utf-8").read().strip().split("\n")
COLORS = [(0, 255, 0), (0, 0, 255)] # Green for Empty, Red for Occupied

# Queues for thread communication
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image to 640x640 with padding to maintain aspect ratio"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, left, top

def capture_worker(source):
    """Thread for high-speed frame capturing"""
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)
    cap.release()

def inference_worker():
    """Thread for ONNX preprocessing, inference, and postprocessing"""
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            # 1. Preprocessing
            img_lb, r, pad_w, pad_h = letterbox(frame, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_chw = img_norm.transpose(2, 0, 1)
            blob = np.expand_dims(img_chw, 0)

            # 2. Inference
            preds = session.run(None, {input_name: blob})[0]

            # 3. Postprocessing
            preds = np.squeeze(preds).T
            boxes_xywh = preds[:, :4]
            scores = np.max(preds[:, 4:], axis=1)
            class_ids = np.argmax(preds[:, 4:], axis=1)

            # Confidence filtering
            idx = scores > CONF_THRESH
            boxes_xywh, scores, class_ids = boxes_xywh[idx], scores[idx], class_ids[idx]

            final_detections = []
            if len(boxes_xywh) > 0:
                # Coordinate scaling
                x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_w) / r
                y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_h) / r
                w_box = boxes_xywh[:, 2] / r
                h_box = boxes_xywh[:, 3] / r
                
                # NMS
                indices = cv2.dnn.NMSBoxes(
                    np.stack([x1, y1, w_box, h_box], axis=1).tolist(), 
                    scores.tolist(), CONF_THRESH, NMS_THRESH
                )

                if len(indices) > 0:
                    for i in indices.flatten():
                        final_detections.append({
                            "box": [int(x1[i]), int(y1[i]), int(x1[i] + w_box[i]), int(y1[i] + h_box[i])],
                            "score": scores[i],
                            "class_id": class_ids[i]
                        })

            if result_queue.full():
                result_queue.get()
            result_queue.put((frame, final_detections))

def main():
    # Start background threads
    t1 = threading.Thread(target=capture_worker, args=(VIDEO_SOURCE,), daemon=True)
    t2 = threading.Thread(target=inference_worker, daemon=True)
    t1.start()
    t2.start()

    prev_time = 0
    while True:
        if not result_queue.empty():
            frame, detections = result_queue.get()
            
            empty_count = 0
            occupied_count = 0

            # Drawing
            for det in detections:
                box = det["box"]
                cls_id = det["class_id"]
                score = det["score"]
                
                if cls_id == 0: empty_count += 1
                else: occupied_count += 1

                color = COLORS[cls_id] if cls_id < len(COLORS) else (255, 0, 0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                label = f"{CLASSES[cls_id]} {score:.2f}"
                cv2.putText(frame, label, (box[0], box[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Performance stats
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            info = f"FPS: {fps:.1f} | E: {empty_count} | O: {occupied_count}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Multi-threaded YOLOv8 ONNX", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
