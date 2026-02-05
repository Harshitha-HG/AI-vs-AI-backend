from fastapi import APIRouter, UploadFile, File
from transformers import pipeline
from PIL import Image
import io

router = APIRouter()

# ----------------------------------
# Load AI Image Detector Model
# ----------------------------------
image_detector = pipeline(
    "image-classification",
    model="umm-maybe/ai-image-detector"
)

# ----------------------------------
# STEP-1: Image Content Origin Analysis
# ----------------------------------
@router.post("/verify-image")
async def verify_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run AI-generated image detection
        result = image_detector(image)[0]
        ai_prob = int(result["score"] * 100)

        # Decide verdict
        if ai_prob >= 80:
            verdict = "Likely AI-Generated Image"
        elif ai_prob >= 50:
            verdict = "Possibly AI-Generated Image"
        else:
            verdict = "Likely Real Image"

        return {
            "status": "success",
            "content_origin_score": ai_prob,
            "verdict": verdict,
            "insights": "Decision based on visual artifacts and texture patterns"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
