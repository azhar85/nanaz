from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware   # <--- tambahkan


# === 1. Load model ===
MODEL_PATH = "model_nanas_best.h5"  # sesuaikan kalau namanya beda
model = tf.keras.models.load_model(MODEL_PATH)

# === 2. Urutan kelas HARUS sesuai class_indices ===
# {'lewat_matang': 0, 'matang': 1, 'mengkal': 2, 'mentah': 3}
CLASS_NAMES = ["lewat_matang", "matang", "mengkal", "mentah"]

IMG_SIZE = (224, 224)

app = FastAPI(title="API Klasifikasi Kematangan Nanas")

# Izinkan akses dari mana saja (untuk pengembangan lokal)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # boleh dibatasi nanti misal ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array



@app.get("/")
def root():
    return {"message": "API klasifikasi kematangan nanas siap."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        preds = model.predict(img_array)
        probas = preds[0]  # shape (4,)

        top_idx = int(np.argmax(probas))
        top_class = CLASS_NAMES[top_idx]
        top_conf = float(probas[top_idx])

        # Biar lebih ramah, mapping ke label bahasa Indonesia
        label_map = {
            "mentah": "Mentah",
            "mengkal": "Mengkal (setengah matang)",
            "matang": "Matang (siap konsumsi)",
            "lewat_matang": "Lewat matang / mulai busuk"
        }

        return JSONResponse({
            "predicted_class": top_class,
            "predicted_label": label_map.get(top_class, top_class),
            "confidence": top_conf,
            "all_probabilities": {
                CLASS_NAMES[i]: float(probas[i]) for i in range(len(CLASS_NAMES))
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
