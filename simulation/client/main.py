import streamlit as st
import requests
from PIL import Image
import base64
import numpy as np
from io import BytesIO
import time
import cv2

if 'toggle' not in st.session_state:
    st.session_state.toggle = True

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()


def get_data():
    try:
        url = "http://127.0.0.1:8000/display"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            image_data = base64.b64decode(data["image"])
            image = np.array(Image.open(BytesIO(image_data)))

            box_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

            for prediction in data["predictions"]["predictions"]:
                box = prediction["box"]
                center_x = int((box['x1'] + box['x2']) / 2)
                center_y = int((box['y1'] + box['y2']) / 2)
                cv2.circle(box_image, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=10)

            return image, box_image
    except Exception as e:
        print("Failed to get data from the server. Error:", e)
        return None, None


def refresh_data():
    image, box_image = get_data()
    if image is not None:
        st.session_state.image = image
        st.session_state.box_image = box_image
        st.session_state.last_refresh = time.time()
        st.rerun()
    else:
        st.error("Failed to get data from the server. Retrying in 30 seconds...")
        time.sleep(30)
        refresh_data()


st.title("DROPEX")
if 'image' in st.session_state:
    if st.session_state.toggle:
        st.image(st.session_state.image, caption="Captured Image from Master Drone")

    st.image(st.session_state.box_image, caption="Detected Objects")
    st.button("Refresh", st.rerun)
    st.session_state.toggle = st.toggle("Show Captured Image with BBOX")
else:
    refresh_data()

if time.time() - st.session_state.last_refresh > 10:
    refresh_data()
