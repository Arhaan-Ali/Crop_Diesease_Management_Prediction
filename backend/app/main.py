from fastapi import FastAPI, File, UploadFile
from app.model.predictor import predict
from app.model.loader import load_model
from app.model.config import Class_name
from PIL import Image
import io
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names = load_model(model_path="rice_disease_model.pt",
                   num_classes=3)
model.to(device)
model.eval()


@app.get("/")
async def root():
    return {"health check": "ok"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    predicted_class, confidence = predict(model, device, image, Class_name)

    return {
        "predicted class": predicted_class,
        "confidence": round(confidence, 2)
    }