import requests
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import streamlit as st
from streamlit_image_zoom import image_zoom


def show_image_from_url(url, predictions, db):
    response = requests.get(url)
    if response.status_code == 200:

        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
        image_cv = np.array(image)
        image_cv = image_cv[:, :, ::-1]

        image_cv = np.ascontiguousarray(image_cv, dtype=np.uint8)

        for pred in predictions:
            box = pred["box"]
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv,
                        f"Class: {pred['class']} Score: {pred['score'][:5]}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2)

        image_pil = Image.fromarray(image_cv[:, :, ::-1])
        image_zoom(image_pil, mode="scroll", size=(620, 380))
    else:
        st.error('Failed to fetch image.')


def fetch_data(db, username='user_123'):
    ref = db.reference(username)
    data = ref.get()
    if data:
        sorted_data = sorted(data.items(), key=lambda item: item[1]['time'], reverse=True)
        latest_entry = sorted_data[0][1]
        return latest_entry
    else:
        return None
