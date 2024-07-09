import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
from torchvision.ops import box_iou
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Detr(pl.LightningModule):

    def __init__(self,
                 lr,
                 lr_backbone,
                 weight_decay,
                 checkpoint,
                 id2label,
                 train_dataloader,
                 val_dataloader):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=checkpoint,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.map_metric = MeanAveragePrecision()

        self.all_preds = []
        self.all_targets = []
        self.iou_threshold = 0.5

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict, outputs

    def training_step(self, batch, batch_idx):
        loss, loss_dict, _ = self.common_step(batch, batch_idx)
        wandb.log({"training_loss": loss, **{"train_" + k: v.item() for k, v in loss_dict.items()}})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step(batch, batch_idx)

        for i in range(outputs.pred_boxes.shape[0]):
            pred = {
                'boxes': outputs.pred_boxes[i].cpu(),
                'scores': outputs.logits[i].softmax(-1)[:, :-1].max(-1)[0].cpu(),
                'labels': outputs.logits[i].softmax(-1)[:, :-1].argmax(-1).cpu()
            }
            self.all_preds.append(pred)

        for t in batch["labels"]:
            target = {
                'boxes': t['boxes'].cpu(),
                'labels': t['class_labels'].cpu()
            }
            self.all_targets.append(target)

        return loss

    def on_validation_epoch_end(self):
        metric_dict = self.map_metric.compute()

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, target in zip(self.all_preds, self.all_targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            true_boxes = target['boxes']
            true_labels = target['labels']

            sorted_indices = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_indices]
            pred_labels = pred_labels[sorted_indices]

            for i, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
                ious = box_iou(box.unsqueeze(0), true_boxes[true_labels == label])

                if ious.shape[1] > 0:
                    max_iou, max_index = ious.max(1)
                    if max_iou > self.iou_threshold:
                        total_tp += 1

                        true_boxes = torch.cat([true_boxes[:max_index], true_boxes[max_index + 1:]])
                        true_labels = torch.cat([true_labels[:max_index], true_labels[max_index + 1:]])
                    else:
                        total_fp += 1
                else:
                    total_fp += 1

            total_fn += len(true_boxes)

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        wandb.log({
            "validation/mAP": metric_dict['map'],
            "validation/mAP_50": metric_dict['map_50'],
            "validation/mAP_75": metric_dict['map_75'],
            "validation/precision": precision,
            "validation/recall": recall,
            "validation/f1": f1
        })

        self.all_preds = []
        self.all_targets = []
        self.map_metric.reset()

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader
