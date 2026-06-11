"""
STRIDON ML Backend
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.routes import router
from config import MODEL_PATH, DATASET_PATH, DATA_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-train on startup if no model exists."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️  No trained model found. Auto-training with synthetic dataset...")
        from ml.dataset_generator import generate_dataset
        from ml.trainer import train

        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(DATASET_PATH):
            df = generate_dataset(600)
            df.to_csv(DATASET_PATH, index=False)
            print(f"✅ Synthetic dataset generated: {len(df)} rows")

        metrics = train(DATASET_PATH)
        print(f"✅ Model trained. Test accuracy: {metrics['test_accuracy']:.4f}")
    else:
        print("✅ Pre-trained model found. Ready to serve.")

    yield  # App runs here

    print("👋 STRIDON backend shutting down.")


app = FastAPI(
    title="STRIDON ML Backend",
    description=(
        "AI-powered stream recommendation and student guidance engine. "
        "Provides Science / Commerce / Arts recommendations with detailed "
        "reasoning derived from ML model internals — no external AI APIs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Replace with your Next.js domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["System"])
def root():
    return {
        "message": "STRIDON ML Backend is running 🚀",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "POST /api/v1/predict",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
