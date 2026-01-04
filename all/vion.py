import cv2
import numpy as np
import threading
import queue
import time
import openvino as ov

# --- SYSTEM CONFIGURATION ---
MODEL_PATH = "best/best.xml"  # Path to your OpenVINO .xml file
VIDEO_PATH = "1230.mp4" # Path to input video or 0 for webcam
IMG_SIZE = 640                # Target size for the model
CONF_THRESHOLD = 0.50         # Increased to 0.5 to filter out background noise
NMS_THRESHOLD = 0.35          # Lowered to 0.35 to strictly remove overlapping boxes

# Class names and colors
CLASS_NAMES = ["Empty Slot", "Occupied"] 
COLORS = [(0, 255, 0), (0, 0, 255)] # [Green, Red]

# Queues for thread communication
raw_frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Standard Letterbox: Resize image while maintaining aspect ratio using padding.
    This prevents image distortion and improves accuracy significantly.
    """
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

def video_capture_worker():
    """ Thread 1: Continuously capture frames from video source """
    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if raw_frame_queue.full():
            raw_frame_queue.get()
        raw_frame_queue.put(frame)
    cap.release()

def inference_worker():
    """ Thread 2: Preprocessing (Letterbox) -> OpenVINO Inference -> Postprocessing (NMS) """
    # Initialize OpenVINO
    core = ov.Core()
    model = core.read_model(MODEL_PATH)
    compiled_model = core.compile_model(model, "CPU")
    output_layer = compiled_model.output(0)

    while True:
        if not raw_frame_queue.empty():
            frame = raw_frame_queue.get()
            
            # --- STEP 1: PREPROCESSING ---
            # Use letterbox to keep aspect ratio (avoids stretching)
            img_lb, r, pad_w, pad_h = letterbox(frame, (IMG_SIZE, IMG_SIZE))
            # Convert BGR (OpenCV) to RGB (YOLO requirement)
            img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            # Normalize and transpose (HWC to CHW)
            blob = img_rgb.astype(np.float32) / 255.0
            blob = blob.transpose((2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            # --- STEP 2: INFERENCE ---
            results = compiled_model([blob])[output_layer]
            
            # --- STEP 3: POSTPROCESSING ---
            # Transform output from [1, 84, 8400] to [8400, 84]
            predictions = np.squeeze(results).T
            boxes_xywh = predictions[:, :4]
            scores = np.max(predictions[:, 4:], axis=1)
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            # Filter by confidence
            mask = scores > CONF_THRESHOLD
            boxes_xywh, scores, class_ids = boxes_xywh[mask], scores[mask], class_ids[mask]

            final_detections = []
            if len(boxes_xywh) > 0:
                # Rescale coordinates considering letterbox padding and ratio
                x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_w) / r
                y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_h) / r
                w_box = boxes_xywh[:, 2] / r
                h_box = boxes_xywh[:, 3] / r
                
                # Apply OpenCV NMS for clean detections
                indices = cv2.dnn.NMSBoxes(
                    np.stack([x1, y1, w_box, h_box], axis=1).tolist(), 
                    scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD
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

def main_display():
    """ Thread 3: UI Rendering and Statistics """
    t_cap = threading.Thread(target=video_capture_worker, daemon=True)
    t_inf = threading.Thread(target=inference_worker, daemon=True)
    t_cap.start()
    t_inf.start()

    prev_time = 0
    while True:
        if not result_queue.empty():
            frame, detections = result_queue.get()
            
            e_count, o_count = 0, 0

            # --- DRAWING RESULTS ---
            for det in detections:
                box, score, cls_id = det["box"], det["score"], det["class_id"]
                if cls_id == 0: e_count += 1
                else: o_count += 1

                color = COLORS[cls_id] if cls_id < len(COLORS) else (255, 0, 0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
                cv2.putText(frame, label, (box[0], box[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Performance monitor
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            info = f"OV FPS: {fps:.1f} | E: {e_count} | O: {o_count}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("OpenVINO Optimized Pipeline", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_display()
