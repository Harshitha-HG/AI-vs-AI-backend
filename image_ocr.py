from fastapi import APIRouter, UploadFile, File
from PIL import Image
import pytesseract
import io
from text_verify import verify_text_logic

router = APIRouter()

# -----------------------------
# Extract text only (unchanged behavior)
# -----------------------------
@router.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        extracted_text = pytesseract.image_to_string(image)

        return {
            "status": "success",
            "extracted_text": extracted_text.strip()
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# -----------------------------
# OCR + Text Accuracy (NEW FLOW)
# -----------------------------
@router.post("/verify-image-text")
async def verify_image_text(file: UploadFile = File(...)):
    try:
        # Read image safely
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1️⃣ OCR
        extracted_text = pytesseract.image_to_string(image).strip()

        if not extracted_text:
            return {
                "status": "error",
                "message": "No readable text found in image"
            }

        # 2️⃣ Reuse text verification logic
        text_result = verify_text_logic(extracted_text)

        return {
            "status": "success",
            "extracted_text": extracted_text,
            "analysis": text_result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
