import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailRequest(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_id = "OliverArt5500/klasyfikatorspamu1"
    
    logger.info("ładowanie modelu")
    app.state.spam_classifier = pipeline("text-classification", model=model_id, tokenizer=model_id)
    logger.info("Model gotowy")

    yield

    logger.info("wyłączanie API")
    app.state.spam_classifier = None

app = FastAPI(title="Klasyfikator spamu", lifespan=lifespan)

@app.post("/predict")
async def predict_spam(request: Request, payload: EmailRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="wiadomość nie może być pusta.")

    try:
        #pobranie modelu
        classifier = request.app.state.spam_classifier
        
        # Wykonujemy predykcję
        prediction = classifier(payload.text, truncation=True, max_length=512)[0]

        return {
            "label": prediction["label"],
            "confidence_score": prediction["score"]
        }
    except Exception as e:
        #Będzie w dockerlogs
        logger.error(f"Krytyczny błąd podczas analizy tekstu: {str(e)}")
        
        raise HTTPException(status_code=500, detail="Wystąpił błąd")
