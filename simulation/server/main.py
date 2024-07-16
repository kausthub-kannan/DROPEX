from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image
import io
import json

from detr import DetectionTransformer
from yolo import YOLO
from fastapi.staticfiles import StaticFiles
from db import upload_data

app = FastAPI()

config = {
    "image_processor_checkpoint": "facebook/detr-resnet-50",
    "model_checkpoint": "./models/detr/model.safetensors",
    "config_path": "./models/detr/config.json",
    "device": "cpu"
}
detr_model = DetectionTransformer(config)
yolo_model = YOLO()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/webgl", StaticFiles(directory="../webgl", html=True), name="webgl")


@app.get("/webgl")
async def root():
    return FileResponse('../webgl/index.html')


@app.post("/stream")
async def stream(snapshot: str = Form(...), time: str = Form(...)):
    img_bytes = base64.b64decode(snapshot)
    img_buffer = io.BytesIO(img_bytes)
    img = Image.open(img_buffer)

    predictions, image = yolo_model.predict(img)
    # predictions, image = detr_model.predict(img)

    db_response = upload_data(predictions, image, time)

    return {
        "message": "Data received successfully",
        "status": True,
        "successfully_uploaded": db_response
        }


@app.get("/display")
async def display():
    with open('database/output.png', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    with open('database/predictions.json', 'r') as file:
        predictions = json.load(file)

    for prediction in predictions["predictions"]:
        prediction["class"] = int(prediction["class"])
        prediction["score"] = float(prediction["score"])
        prediction["box"]["x1"] = int(prediction["box"]["x1"])
        prediction["box"]["y1"] = int(prediction["box"]["y1"])
        prediction["box"]["x2"] = int(prediction["box"]["x2"])
        prediction["box"]["y2"] = int(prediction["box"]["y2"])

    return JSONResponse(content={"image": encoded_image, "predictions": predictions})
