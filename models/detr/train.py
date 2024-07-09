import os
from transformers import DetrForObjectDetection, DetrImageProcessor
from dataloader import CocoDetection
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from model import Detr
from utils import get_config
from evaluation import evaluate
from dotenv import load_dotenv
import logging

load_dotenv()
os.environ["WANDDB_API_KEY"] = os.getenv("WANDB_API_KEY")
logger = logging.getLogger("train.py")
logger.setLevel(logging.INFO)
config = get_config()

logger.info("Fetching DeTr model ....\n\n")
image_processor = DetrImageProcessor.from_pretrained(config["checkpoint"])
model = DetrForObjectDetection.from_pretrained(config["checkpoint"])
model.to(config["device"])
logger.info("Model loaded!")

train_data = CocoDetection("train", image_processor, root)
val_data = CocoDetection("valid", image_processor, root)
test_data = CocoDetection("test", image_processor, root)
logger.info("Data loaded")

categories = train_data.coco.cats
id2label = {k: v['name'] for k, v in categories.items()}


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

logger.info("Creating Data loaders")
train_dataloader = DataLoader(dataset=train_data,
                              collate_fn=collate_fn,
                              batch_size=config["batch_size"],
                              num_workers=config["num_workers"],
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_data,
                            collate_fn=collate_fn,
                            batch_size=config["batch_size"],
                            num_workers=config["num_workers"],
                            shuffle=False)

test_dataloader = DataLoader(dataset=test_data,
                             collate_fn=collate_fn,
                             batch_size=config["batch_size"],
                             num_workers=config["num_workers"],
                             shuffle=False)
logger.info("Data loaders created")

model = Detr(lr=config["lr"],
             lr_backbone=config["lr_backbone"],
             weight_decay=config["weight_decay"],
             checkpoint=config["checkpoint"],
             id2label=id2label
             )

batch = next(iter(train_dataloader))
outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

wandb.init(config["wandb_project_name"])
wandb_logger = WandbLogger(log_model="all")

logger.info("Model training started ...")
trainer = Trainer(logger=wandb_logger,
                  devices=config["number_of_devices"],
                  accelerator="gpu",
                  max_epochs=config["epochs"],
                  gradient_clip_val=config["gradient_clip_val"],
                  accumulate_grad_batches=config["accumulate_grad_batches"],
                  log_every_n_steps=config["log_every_n_steps"],
                  )
trainer.fit(model)
logger.info(f"Model trained successfully for" + config["number_of_devices"] + "epochs sucessfuly. \n Logs have been saved to wandb dashboard")

wandb.finish()

model.to(config["device"])
model.save_pretrained(config["output_checkpoint"])

evaluate(test_data, test_dataloader, model, image_processor, config)
