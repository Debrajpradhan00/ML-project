from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ✅ FIX 1: Removed unused StaticFiles import (caused errors if
#           python-multipart wasn't installed correctly)

# ✅ FIX 2: Wrap model import so server gives a clear error
#           instead of crashing silently if .pkl files are missing
try:
    from lungmodel import predict_image
    MODEL_READY = True
except FileNotFoundError as e:
    print(f"\n❌ {e}")
    MODEL_READY = False

app = FastAPI(title="Lung Cancer Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "Lung Cancer Detection API is running ✅",
        "model_ready": MODEL_READY,
        "message": "Run python main.py first if model_ready is false"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ✅ FIX 3: Clear error if model not trained yet
    if not MODEL_READY:
        raise HTTPException(
            503,
            "Model not loaded. Run 'python main.py' first to train and save the model."
        )

    allowed = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(400, f"Only these formats accepted: {', '.join(allowed)}")

    img_bytes = await file.read()

    # ✅ FIX 4: Check file isn't empty
    if len(img_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        result = predict_image(img_bytes)
        return result
    except ValueError as e:
        raise HTTPException(422, f"Image processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    