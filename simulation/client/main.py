import streamlit as st
import firebase_admin
from firebase_admin import credentials, db

from components import search_bar
from utils import show_image_from_url

try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate("dropex-2024-firebase-adminsdk.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://dropex-2024-default-rtdb.asia-southeast1.firebasedatabase.app",
        "storageBucket": "dropex-2024.appspot.com",
    })

database = db.reference()


st.title("DROPEX")
col1, col2 = st.columns([10, 3])

data = search_bar(db)

if data:
    show_image_from_url(data['image_url'], data['predictions'], db)
else:
    st.write("Enter your Username (same as in simulation) to view predictions")
