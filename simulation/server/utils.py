import numpy as np
import cv2
from PIL import Image

class_names = ['Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare']

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

conf_threshold = 0.6
iou_threshold = 0.3
__class_names = ['person']


def process_output(output):
    predictions = np.squeeze(output[0]).T

    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], []

    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = extract_boxes(predictions)
    indices = multiclass_nms(boxes, scores, class_ids, iou_threshold)

    return boxes[indices], scores[indices], class_ids[indices]


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def extract_boxes(predictions):
    boxes = predictions[:, :4]
    boxes = rescale_boxes(boxes)
    boxes = xywh2xyxy(boxes)

    return boxes


def rescale_boxes(boxes):
    input_shape = np.array([640] * 4)
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([1920, 1080, 1920, 1080])
    return boxes


def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = np.stack((gray_img,) * 3, axis=-1)
    return gray_image


def thermal_mapping(img):
    gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    thermal_image = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
    thermal_pil = Image.fromarray(thermal_image)
    return np.array(thermal_pil)
