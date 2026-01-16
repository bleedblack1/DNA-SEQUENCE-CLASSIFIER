from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import logging
import sys
from pathlib import Path
from threading import Lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deployment.predict import PredictionPipeline  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=" DNA Sequence Classification API",
    description="Predict DNA sequence functions using trained ML models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    pipeline = PredictionPipeline()
    logger.info(" Prediction pipeline loaded")
except Exception as e:
    logger.exception(" Failed to load prediction pipeline")
    pipeline = None

# GPU / inference safety
model_lock = Lock()



class PredictionRequest(BaseModel):
    sequence: str
    model: str = "cnn"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sequence": "ATGATGATGATGATGATG",
                "model": "cnn"
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    sequences: list[str]
    model: str = "cnn"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sequences": [
                    "ATGATGATGATGATGATG",
                    "GCTAGCTAGCTAGCTAGC"
                ],
                "model": "cnn"
            }
        }
    )


class PredictionResponse(BaseModel):
    sequence_length: int
    model_used: str
    prediction: str
    confidence: float
    probabilities: dict



def validate_model(model: str):
    available = ["cnn"] + list(pipeline.baseline_models.keys())
    if model not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be one of {available}"
        )



@app.get("/health")
def health():
    if pipeline is None:
        return {"status": "error", "message": "Pipeline not loaded"}

    return {
        "status": "healthy",
        "cnn_loaded": pipeline.cnn_model is not None,
        "baseline_models": list(pipeline.baseline_models.keys())
    }



@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not loaded")

    validate_model(request.model)

    if not request.sequence:
        raise HTTPException(status_code=400, detail="Sequence cannot be empty")

    with model_lock:
        result = pipeline.predict(request.sequence, use_model=request.model)

    return result



@app.post("/batch_predict")
def batch_predict(request: BatchPredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not loaded")

    validate_model(request.model)

    if not request.sequences:
        raise HTTPException(status_code=400, detail="Sequences list cannot be empty")

    results = []
    with model_lock:
        for seq in request.sequences:
            results.append(pipeline.predict(seq, use_model=request.model))

    return {
        "total_sequences": len(results),
        "model_used": request.model,
        "predictions": results
    }



@app.post("/predict_from_fasta")
async def predict_from_fasta(
    file: UploadFile = File(...),
    model: str = "cnn"
):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not loaded")

    validate_model(model)

    contents = (await file.read()).decode("utf-8")

    sequences = []
    seq_ids = []
    current_id = None
    current_seq = ""

    for line in contents.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_seq:
                sequences.append(current_seq)
                seq_ids.append(current_id)
            current_id = line[1:]
            current_seq = ""
        else:
            current_seq += line

    if current_seq:
        sequences.append(current_seq)
        seq_ids.append(current_id)

    results = []
    with model_lock:
        for sid, seq in zip(seq_ids, sequences):
            r = pipeline.predict(seq, use_model=model)
            r["sequence_id"] = sid
            results.append(r)

    return {
        "total_sequences": len(results),
        "model_used": model,
        "predictions": results
    }



@app.get("/models")
def models():
    return {
        "available_models": ["cnn"] + list(pipeline.baseline_models.keys())
    }
