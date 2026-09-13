from pathlib import Path
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.vit_service import analyze_image

app = FastAPI(
    title="SignalScope Forensic Engine",
    description="Media Authenticity & Interpretability Verification Engine",
    version="2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Serve static web frontend
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "SignalScope Engine Online", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Vision Transformer (ViT-B/16)",
        "benchmark_auc": 0.9937,
        "benchmark_f1": 0.9617
    }

@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    simulate_jpeg: bool = Form(False)
):
    try:
        # Save temporary uploaded file
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            # Run the multi-signal forensic evaluation pipeline
            results = analyze_image(tmp_path, simulate_jpeg=simulate_jpeg)
            results["filename"] = file.filename
            return JSONResponse(content=results)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
