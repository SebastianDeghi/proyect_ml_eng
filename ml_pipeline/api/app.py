"""
app.py - FastAPI application for sentiment analysis

This module provides a REST API endpoint for sentiment prediction.

Usage:
    uvicorn app:app --reload

Endpoints:
    GET  /            - Health check
    GET  /health      - Health check
    POST /predict     - Sentiment prediction
    POST /batch       - Batch sentiment prediction
"""

import sys
import os

# Add parent directory to path to import predict module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn

# Import prediction module
from predict import load_model_and_vectorizer, predict_sentiment, predict_batch

# =============================================
# Pydantic Models (Request/Response Schemas)
# =============================================

class PredictRequest(BaseModel):
    """Request schema for single prediction."""
    text: str = Field(..., description="Movie review text to analyze", min_length=1, max_length=10000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely amazing! I loved every minute of it."
            }
        }


class PredictResponse(BaseModel):
    """Response schema for single prediction."""
    sentiment: str = Field(..., description="Predicted sentiment: 'positive' or 'negative'")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)
    text_length: int = Field(..., description="Length of input text in characters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.9876,
                "text_length": 42
            }
        }


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction."""
    texts: List[str] = Field(..., description="List of movie reviews to analyze", min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "This movie was great!",
                    "Terrible film, waste of time."
                ]
            }
        }


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction."""
    results: List[PredictResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions processed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {"sentiment": "positive", "confidence": 0.95, "text_length": 20},
                    {"sentiment": "negative", "confidence": 0.92, "text_length": 28}
                ],
                "total_count": 2
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }


# =============================================
# FastAPI Application
# =============================================

# Create FastAPI app with metadata
app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="""
    ## Sentiment Analysis API for IMDB Movie Reviews
    
    This API uses a **TF-IDF + Logistic Regression** model trained on 50,000 IMDB reviews.
    
    ### Features:
    - Single text prediction
    - Batch prediction (up to 100 texts)
    - Confidence scores for each prediction
    - Real-time inference (~15ms per request)
    
    ### Model Performance:
    - Accuracy: 89.4%
    - F1-score: 89.5%
    - Precision: 89.6%
    - Recall: 89.5%
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (allow frontend applications to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model caching
_model = None
_vectorizer = None


# =============================================
# Startup Event: Load Model
# =============================================

@app.on_event("startup")
async def startup_event():
    """Load model and vectorizer on application startup."""
    global _model, _vectorizer
    print("Loading sentiment analysis model...")
    _model, _vectorizer = load_model_and_vectorizer()
    print("Model loaded successfully!")


# =============================================
# Health Check Endpoints
# =============================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify API is running."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        version="1.0.0"
    )


# =============================================
# Prediction Endpoints
# =============================================

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict sentiment of a single movie review.
    
    Args:
        request: JSON object with 'text' field containing the review.
    
    Returns:
        PredictResponse with sentiment, confidence, and text length.
    
    Raises:
        HTTPException: If prediction fails or model not loaded.
    """
    if _model is None or _vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Get prediction
        result = predict_sentiment(request.text, _model, _vectorizer)
        
        return PredictResponse(
            sentiment=result['sentiment'],
            confidence=round(result['confidence'], 4),
            text_length=len(request.text)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictRequest):
    """
    Predict sentiment for multiple movie reviews (batch mode).
    
    Args:
        request: JSON object with 'texts' list of reviews.
    
    Returns:
        BatchPredictResponse with list of predictions.
    
    Raises:
        HTTPException: If prediction fails or model not loaded.
    """
    if _model is None or _vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        results = []
        for text in request.texts:
            result = predict_sentiment(text, _model, _vectorizer)
            results.append(PredictResponse(
                sentiment=result['sentiment'],
                confidence=round(result['confidence'], 4),
                text_length=len(text)
            ))
        
        return BatchPredictResponse(
            results=results,
            total_count=len(results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# =============================================
# Root endpoint with documentation redirect
# =============================================

@app.get("/info", tags=["Info"])
async def get_info():
    """Get model information and performance metrics."""
    return {
        "model_type": "Logistic Regression",
        "representation": "TF-IDF",
        "vocabulary_size": 10000,
        "metrics": {
            "accuracy": 0.894,
            "precision": 0.896,
            "recall": 0.895,
            "f1_score": 0.8955
        },
        "language": "english",
        "task": "binary_sentiment_classification"
    }


# =============================================
# Main entry point
# =============================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )