import numpy as np
from PIL import Image
import cv2
import onnxruntime as rt
from utils import grayscale, extract_boxes, multiclass_nms, thermal_mapping


class YOLO:
    def __init__(self):
        self.output = None
        self.sess = rt.InferenceSession("models/pdti_and_unirid.onnx")

    def predict(self, img):
        """
        Predicts the bounding boxes and classes of the objects in the image using YOLOv8.

        :param img: PIL.Image.Image

        :return: predictions: list - List of dictionary of format {"class": str, "score": str, "box": dict}
                 img: np.array - Pseudo Thermal Image in numpy array format
        """
        gray_image = grayscale(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        image = Image.fromarray(gray_image)
        img_resize = image.resize((640, 640))
        img_np = np.array(img_resize)
        img_np = img_np.transpose((2, 0, 1))
        img_np = img_np.astype("float32") / 255.0
        img_np = np.expand_dims(img_np, axis=0)

        self.output = self.sess.run(None, {self.sess.get_inputs()[0].name: img_np})
        boxes, scores, class_ids = self._post_process()
        predictions = self._to_dictionary(boxes, scores, class_ids)

        return predictions, thermal_mapping(img)

    def _post_process(self, iou_threshold=0.3, conf_threshold=0.6):
        """
        Post-process the output of the model to get the bounding boxes, scores, and class IDs.

        :param iou_threshold: float - Intersection over Union threshold for non-maximum suppression
        :param conf_threshold: float - confidence threshold for filtering out the predictions

        :return: boxes: list, scores: list, class_ids: list - Bounding boxes, scores, and class IDs of the predictions
        """
        predictions = np.squeeze(self.output[0]).T

        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = extract_boxes(predictions)
        indices = multiclass_nms(boxes, scores, class_ids, iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def _to_dictionary(self, boxes, scores, classes):
        """
        Convert the results to a list of dictionary of format {"class": str, "score": str, "box": dict}

        :param boxes: list - Bounding boxes
        :param scores: list - Scores
        :param classes: list - Class IDs

        :return: predictions: list - List of dictionary of format {"class": str, "score": str, "box": dict}
        """
        predictions = []
        for box, score, class_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = (
                int(box[0]),
                int(box[1]) - 60,
                int(box[2]),
                int(box[3]) - 50,
            )

            predictions.append(
                {
                    "class": str(class_id),
                    "score": str(score * 100),
                    "box": {"x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2)},
                }
            )

        return predictions
