import cv2
import numpy as np
import onnxruntime as ort

# ===== CONFIG =====
MODEL_PATH = "best3.onnx"
CLASSES_PATH = "classes.txt"
IMG_SIZE = 640
CONF_THRESH = 0.1
NMS_THRESH = 0.45

# ===== LOAD MODEL =====
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASSES = [x.strip() for x in f.read().splitlines() if x.strip()]

# ===== LETTERBOX (returns integer paddings) =====
def letterbox(img, new_shape=640, color=(114,114,114)):
    """
    Resize and pad image to new_shape (int square) while keeping aspect ratio.
    Returns padded image, scale ratio r, left pad, top pad (all integers).
    """
    h0, w0 = img.shape[:2]
    if isinstance(new_shape, int):
        new_w = new_h = new_shape
    else:
        new_w, new_h = new_shape

    # scale ratio
    r = min(new_w / w0, new_h / h0)
    resized_w, resized_h = int(round(w0 * r)), int(round(h0 * r))

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h

    # integer padding split (left, right, top, bottom)
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, left, top

# ===== PREPROCESS =====
def preprocess(img):
    orig = img.copy()
    img_lb, r, left, top = letterbox(img, (IMG_SIZE, IMG_SIZE))
    img_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_lb = img_lb.astype(np.float32) / 255.0
    img_lb = img_lb.transpose(2, 0, 1)  # CHW
    img_lb = np.expand_dims(img_lb, 0)  # 1CHW
    return img_lb, orig, r, left, top

# ===== POSTPROCESS with per-class NMS =====
def postprocess(preds, orig, r, left, top, conf_thres=CONF_THRESH, nms_thres=NMS_THRESH):
    """
    preds: raw model output (any of shapes found) -> normalized to (N, C_out)
           expected content per-row: [cx, cy, w, h, cls0_prob, cls1_prob, ...]
    Returns: annotated image, final_boxes, final_scores, final_class_ids
    """
    # Colors
    COLORS = [(0,255,0), (0,0,255)]
    h_img, w_img = orig.shape[:2]

    p = preds
    # normalize shape to (N, D)
    # common possibilities: (1,6,8400) or (1,8400,6) or (6,8400) etc.
    if p.ndim == 3 and p.shape[1] == 6:
        p = p[0].T        # (8400,6)
    elif p.ndim == 3 and p.shape[2] == 6:
        p = p[0]          # (8400,6)
    elif p.ndim == 2 and p.shape[1] == 6:
        p = p             # already (8400,6)
    else:
        # fallback: try to flatten first dimension if it's 1
        if p.ndim == 3 and p.shape[0] == 1:
            p = p.squeeze(0)
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}")

    # Ensure float
    p = p.astype(np.float32)

    # first 4 columns are bbox cx,cy,w,h; remaining are class probs (any number)
    if p.shape[1] < 5:
        raise ValueError("Model output has fewer than 5 channels: cannot parse.")
    boxes_xywh = p[:, :4]               # (M,4)
    class_probs = p[:, 4:]              # (M, num_classes)
    num_classes = class_probs.shape[1]

    # scores and class ids from class_probs (no separate objectness here)
    scores_all = np.max(class_probs, axis=1)
    class_ids_all = np.argmax(class_probs, axis=1)

    # filter by confidence
    keep_mask = scores_all > conf_thres
    if not np.any(keep_mask):
        return orig, [], [], []

    boxes_xywh = boxes_xywh[keep_mask]
    scores_all = scores_all[keep_mask]
    class_ids_all = class_ids_all[keep_mask]

    # convert xywh (center) -> xyxy
    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    bw = boxes_xywh[:, 2]
    bh = boxes_xywh[:, 3]

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # scale back to original image coordinates (remove pad, divide by scale)
    boxes_xyxy[:, [0,2]] -= left
    boxes_xyxy[:, [1,3]] -= top
    boxes_xyxy /= r

    # clip to image
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w_img - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h_img - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w_img - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h_img - 1)

    # prepare per-class NMS: for each class, run NMS on boxes of that class
    final_boxes = []
    final_scores = []
    final_classes = []

    for cls in range(num_classes):
        # indices of this class
        inds = np.where(class_ids_all == cls)[0]
        if inds.size == 0:
            continue

        boxes_cls = boxes_xyxy[inds]  # shape (k,4)
        scores_cls = scores_all[inds]

        # convert to x,y,w,h for cv2 NMS
        boxes_xywh_for_nms = boxes_cls.copy()
        boxes_xywh_for_nms[:, 2] = boxes_xywh_for_nms[:, 2] - boxes_xywh_for_nms[:, 0]  # w = x2 - x1
        boxes_xywh_for_nms[:, 3] = boxes_xywh_for_nms[:, 3] - boxes_xywh_for_nms[:, 1]  # h = y2 - y1

        # cv2.dnn.NMSBoxes expects lists
        boxes_list = boxes_xywh_for_nms.tolist()
        scores_list = scores_cls.tolist()

        idxs = cv2.dnn.NMSBoxes(boxes_list, scores_list, conf_thres, nms_thres)
        if len(idxs) == 0:
            continue

        idxs = np.array(idxs).reshape(-1)
        for i in idxs:
            sel = inds[i]  # index in the post-filter arrays
            x1i, y1i, x2i, y2i = boxes_xyxy[sel]
            # integer coords
            xi1 = int(round(x1i))
            yi1 = int(round(y1i))
            xi2 = int(round(x2i))
            yi2 = int(round(y2i))

            final_boxes.append([xi1, yi1, xi2, yi2])
            final_scores.append(float(scores_all[sel]))
            final_classes.append(int(cls))

    # draw results
    for bbox, scr, cls in zip(final_boxes, final_scores, final_classes):
        x1, y1, x2, y2 = bbox
              
        color = COLORS[cls] if cls < len(COLORS) else (0, 255, 0)
               
        cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
        
        # label = f"{CLASSES[cls]} {scr:.2f}"
        # cv2.putText(orig, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return orig, final_boxes, final_scores, final_classes


# ===== RUN =====
if __name__ == "__main__":
    img = cv2.imread("t8.jpg")
    if img is None:
        raise SystemExit("Image not found: t5.jpg")

    blob, orig, r, left, top = preprocess(img)
    preds_all = session.run(None, {input_name: blob})
    preds = preds_all[0]

    print("MODEL OUTPUT SHAPE:", preds.shape)

    result_img, boxes, scores, ids = postprocess(preds, orig, r, left, top)

    print("TOTAL DETECTIONS:", len(boxes))
    for i in range(len(boxes)):
        clsname = CLASSES[ids[i]] if ids[i] < len(CLASSES) else str(ids[i])
        print(f"- {clsname}: {scores[i]:.2f}, {boxes[i]}")

    cv2.imwrite("result_fixed8.jpg", result_img)
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
