import firebase_admin
from firebase_admin import credentials, db, storage
import cv2
import uuid

cred = credentials.Certificate("dropex-2024-firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://dropex-2024-default-rtdb.asia-southeast1.firebasedatabase.app",
    "storageBucket": "dropex-2024.appspot.com",
})

database = db.reference()
bucket = storage.bucket()


def upload_file(image):
    try:
        filename = f"{uuid.uuid4()}.png"
        _, img_encoded = cv2.imencode('.png', image)
        img_bytes = img_encoded.tobytes()

        bucket = storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_string(img_bytes, content_type='image/png')
        blob.make_public()

        file_url = blob.public_url
        return file_url
    except Exception as e:
        print("File Upload Error:" + str(e))


def upload_data(predictions, image, time):
    try:
        image_url = upload_file(image)

        database.child('user_123').push({
            'predictions': predictions,
            'time': time,
            'image_url': image_url
        })

        return True
    except Exception as e:
        print("Data Upload error:" + str(e))
