from transformers import DetrImageProcessor
from transformers import AutoConfig, AutoModelForObjectDetection
import torch
from PIL import Image
import numpy as np
from utils import grayscale, thermal_mapping
import cv2


class DetectionTransformer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config["device"])
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.config["model_checkpoint"],
            config=AutoConfig.from_pretrained(self.config["config_path"])
        ).to(self.device)

        self.image_processor = DetrImageProcessor.from_pretrained(self.config["image_processor_checkpoint"])

    def _detect(self, image_np):
        image = Image.fromarray(image_np)
        inputs = self.image_processor(images=image, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        outputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}

        out_logits, out_bbox = outputs['logits'], outputs['pred_boxes']
        processed_outputs = type('obj', (object,), {'logits': out_logits, 'pred_boxes': out_bbox})

        self.results = self.image_processor.post_process_object_detection(
            processed_outputs,
            threshold=0.74,
            target_sizes=[(image.height, image.width)]
        )

    def _process_results(self):
        if isinstance(self.results, list):
            return [{
                        key: (value.item() if isinstance(value, torch.Tensor) and value.numel() == 1
                              else value.tolist() if isinstance(value, torch.Tensor) else value)
                        for key, value in result.items()
                    } if isinstance(result, dict) else result
                    for result in self.results]
        elif isinstance(self.results, dict):
            return {
                key: (value.item() if isinstance(value, torch.Tensor) and value.numel() == 1
                      else value.tolist() if isinstance(value, torch.Tensor) else value)
                for key, value in self.results.items()
            }
        else:
            return self.results

    def _to_dictionary(self, boxes, scores, classes):
        predictions = []
        for box, score, class_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]) - 60, int(box[2]), int(box[3]) - 50

            predictions.append({
                "class": str(class_id),
                "score": str(score * 100),
                "box": {
                    "x1": str(x1),
                    "y1": str(y1),
                    "x2": str(x2),
                    "y2": str(y2)
                }
            })

        return predictions

    def predict(self, img):
        gray_image = grayscale(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        self._detect(gray_image)

        results = self._process_results()
        predictions = self._to_dictionary(results[0]["boxes"], results[0]["scores"], results[0]["labels"])

        return predictions, thermal_mapping(img)
