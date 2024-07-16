import base64
import io
import logging

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from db import upload_data
from detr import DetectionTransformer
from schema import StreamResponse
from yolo import YOLO

app = FastAPI()
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

config = {
    "image_processor_checkpoint": "facebook/detr-resnet-50",
    "model_checkpoint": "./models/detr/model.safetensors",
    "config_path": "./models/detr/config.json",
    "device": "cpu",
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
    """
    Serve the WebGL application
    """
    logging.info("Serving WebGL application")
    return FileResponse("../webgl/index.html")


@app.post("/stream")
async def stream(snapshot: str = Form(...), time: str = Form(...)):
    """
    Stream the snapshot from the client and process it using the YOLOV8 or DETR model

    :param snapshot: str - Base64 encoded image
    :param time: str - Time of the snapshot
    """
    img_bytes = base64.b64decode(snapshot)
    img_buffer = io.BytesIO(img_bytes)
    img = Image.open(img_buffer)

    logging.info(f"Received snapshot at {time}. Performing object detection.")
    predictions, image = yolo_model.predict(img)
    # predictions, image = detr_model.predict(img)
    logging.info(f"Object detection completed. {len(predictions)} objects detected.")

    db_response = upload_data(predictions, image, time)
    logging.info("Data uploaded to Firebase.")

    response = StreamResponse(
        message="Data received successfully",
        status=True,
        successfully_uploaded=db_response,
    )

    return JSONResponse(content=response.dict())
