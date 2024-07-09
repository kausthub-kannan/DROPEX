from coco_eval import CocoEvaluator
from tqdm import tqdm
import torch
from utils import prepare_for_coco_detection


def evaluate(test_data, test_dataloader, model, image_processor, config):
    evaluator = CocoEvaluator(coco_gt=test_data.coco, iou_types=["bbox"])

    print("Running evaluation...")

    for idx, batch in enumerate(tqdm(test_dataloader)):
        pixel_values = batch["pixel_values"].to(config["device"])
        pixel_mask = batch["pixel_mask"].to(config["device"])
        labels = [{k: v.to(config["device"]) for k, v in t.items()} for t in batch["labels"]]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        evaluator.update(predictions)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    return evaluator
