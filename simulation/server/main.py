import cv2
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import base64
from PIL import Image
import io
import numpy as np
import json
from utils import process_output, grayscale, thermal_mapping
from yolo import model

app = FastAPI()


@app.post("/stream")
async def stream(snapshot: str = Form(...), time: str = Form(...)):
    img_bytes = base64.b64decode(snapshot)
    img_buffer = io.BytesIO(img_bytes)
    img = Image.open(img_buffer)
    gray_image = grayscale(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    output = model(Image.fromarray(gray_image))

    boxes, scores, classes = process_output(np.array(output))
    img_cv = thermal_mapping(img)

    predictions = {"predictions": [], "time": time}
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = int(box[0]), int(box[1])-60, int(box[2]), int(box[3])-50

        predictions["predictions"].append({
            "class": str(class_id),
            "score": str(score * 100),
            "box": {
                "x1": str(x1),
                "y1": str(y1),
                "x2": str(x2),
                "y2": str(y2)
            }
        })

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 255, 255), 3)

    cv2.imwrite('database/output.png', img_cv)

    with open('database/predictions.json', 'w') as file:
        json.dump(predictions, file, indent=4)

    return {"message": "Data received successfully", "status": True}


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
