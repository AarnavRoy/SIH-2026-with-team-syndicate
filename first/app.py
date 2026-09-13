from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0
from torchvision import transforms

from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# ============================================================
# 1. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="SignalScope",
    description="Real vs AI-generated image detector",
    version="1.0"
)


# ============================================================
# 2. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/website")
@app.get("/")
def website():
    index_path = Path(__file__).resolve().parent / "index.html"
    if not index_path.exists():
        index_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(index_path)




# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 4. CREATE EFFICIENTNET-B0
# ============================================================

model = efficientnet_b0(weights=None)

# Same classifier structure used during training
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "efficientnet_cifake_epoch_5.pth"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. CLASS NAMES
# ============================================================

# IMPORTANT:
# Your training dataset had:
#
# FAKE = 0
# REAL = 1

class_names = {
    0: "AI GENERATED",
    1: "REAL"
}


# ============================================================
# 8. HOME PAGE
# ============================================================

@app.get("/api")
@app.get("/health")
def home():

    return {
        "message": "SignalScope API is running",
        "model": "EfficientNet-B0",
        "classes": [
            "AI GENERATED",
            "REAL"
        ]
    }


# ============================================================
# 9. PREDICTION API
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        image_bytes = await file.read()

        image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")


        # ----------------------------------------------------
        # Preprocess image
        # ----------------------------------------------------

        image_tensor = transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()


        # ----------------------------------------------------
        # Get probabilities
        # ----------------------------------------------------

        fake_probability = (
            probabilities[0][0].item()
        )

        real_probability = (
            probabilities[0][1].item()
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = max(
            fake_probability,
            real_probability
        )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = class_names[prediction]


        return {
            "success": True,

            "filename": file.filename,

            "prediction": result,

            "confidence": round(
                confidence * 100,
                2
            ),

            "ai_probability": round(
                fake_probability * 100,
                2
            ),

            "real_probability": round(
                real_probability * 100,
                2
            )
        }


    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# 10. RUN SERVER
# ============================================================

# Run with:
#
# python -m uvicorn app:app --reload
#