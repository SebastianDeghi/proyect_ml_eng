"""
predict.py - Prediction module for sentiment analysis

This module provides functions to predict sentiment from raw text
using the pre-trained TF-IDF vectorizer and Logistic Regression model.

Usage:
    from predict import predict_sentiment, load_model_and_vectorizer
    
    # Load once at application startup
    model, vectorizer = load_model_and_vectorizer()
    
    # Predict a single review
    result = predict_sentiment("This movie is amazing!", model, vectorizer)
    print(result)  # {'sentiment': 'positive', 'confidence': 0.95}
    
    # Predict multiple reviews
    results = predict_batch(["Great film!", "Terrible movie."], model, vectorizer)
"""

import os
import re
import joblib
import yaml
import numpy as np
from typing import Dict, Tuple, Optional, List, Any

# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    This function searches for the config file in multiple locations:
    1. The provided path
    2. Same directory as this script
    3. Parent directory of this script (project root)
    4. Current working directory
    5. Docker-specific paths
    
    Args:
        config_path: Path to the configuration file (default: "config.yaml")
    
    Returns:
        Dictionary containing configuration parameters
    
    Raises:
        FileNotFoundError: If config file cannot be found in any location
    """
    # List of possible paths to search for the config file
    possible_paths = [
        config_path,                                        # Provided path
        os.path.join(os.path.dirname(__file__), config_path),  # Same as predict.py
        os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path),  # Project root
        os.path.join(os.getcwd(), config_path),             # Current working directory
        '/app/config.yaml',                                 # Docker container path
        './config.yaml',                                    # Relative path
        '../config.yaml',                                   # Parent relative path
    ]
    
    # Try each path until we find the config file
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Config loaded from: {path}")
            return config
    
    # If we get here, no config file was found
    raise FileNotFoundError(f"Config file not found. Tried: {possible_paths}")


# =============================================================================
# Global Variables (Lazy Loading)
# =============================================================================

_model = None          # Cached model instance
_vectorizer = None     # Cached vectorizer instance
_config = None         # Cached configuration
_stop_words = None     # Cached set of English stopwords
_lemmatizer = None     # Cached WordNet lemmatizer


def _get_nlp_resources():
    """
    Lazy load NLTK resources (stopwords and lemmatizer).
    
    This function downloads NLTK data on first call and caches the resources
    for subsequent calls to avoid redundant downloads.
    
    Returns:
        Tuple of (stop_words_set, lemmatizer_instance)
    """
    global _stop_words, _lemmatizer
    
    # Only download and initialize if not already done
    if _stop_words is None:
        import nltk
        
        # Download required NLTK data (quiet mode to suppress output)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        # Create set of English stopwords for O(1) lookup
        _stop_words = set(stopwords.words('english'))
        
        # Create WordNet lemmatizer for reducing words to base form
        _lemmatizer = WordNetLemmatizer()
    
    return _stop_words, _lemmatizer


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_vectorizer(
    model_path: Optional[str] = None,
    vectorizer_path: Optional[str] = None
) -> Tuple[object, object]:
    """
    Load the pre-trained model and vectorizer from disk.
    
    This function implements intelligent path searching to locate the model
    and vectorizer files in various possible locations (local development,
    Docker containers, different working directories).
    
    Results are cached globally so subsequent calls return immediately.
    
    Args:
        model_path: Optional custom path to the model file
        vectorizer_path: Optional custom path to the vectorizer file
    
    Returns:
        Tuple of (model, vectorizer) objects
    
    Raises:
        FileNotFoundError: If model or vectorizer files cannot be found
    """
    global _model, _vectorizer, _config
    
    # Return cached instances if already loaded
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer
    
    # Load configuration if not already cached
    if _config is None:
        _config = load_config()
    
    # Get file names from configuration
    model_filename = _config['paths']['model_filename']
    vectorizer_filename = _config['paths']['vectorizer_filename']
    models_dir = _config['paths']['models_dir']
    
    # Determine base directories
    current_dir = os.path.dirname(os.path.abspath(__file__))  # /app/ml_pipeline
    root_dir = os.path.dirname(current_dir)                    # /app
    
    # Debug output (helpful for troubleshooting)
    print(f"Looking for model: {model_filename}")
    print(f"Models directory: {models_dir}")
    print(f"Current directory: {current_dir}")
    print(f"Root directory: {root_dir}")
    
    # ========================================================================
    # Build comprehensive list of possible model paths
    # ========================================================================
    possible_model_paths = [
        # User-provided path
        model_path,
        
        # Path from config (ml_pipeline/models/model.pkl)
        os.path.join(models_dir, model_filename),
        
        # Absolute path from root
        os.path.join(root_dir, models_dir, model_filename),
        
        # Within ml_pipeline directory (legacy location)
        os.path.join(current_dir, model_filename),
        os.path.join(root_dir, 'ml_pipeline', model_filename),
        
        # Within models subdirectory
        os.path.join(current_dir, 'models', model_filename),
        
        # Docker container specific paths
        f"/app/{models_dir}/{model_filename}",
        '/app/ml_pipeline/models/model.pkl',
        '/app/ml_pipeline/model.pkl',
        '/app/model.pkl',
        
        # Relative paths
        f"./{model_filename}",
        f"../{model_filename}",
        f"../ml_pipeline/{model_filename}",
        f"../ml_pipeline/models/{model_filename}",
    ]
    
    # ========================================================================
    # Build comprehensive list of possible vectorizer paths
    # ========================================================================
    possible_vectorizer_paths = [
        # User-provided path
        vectorizer_path,
        
        # Path from config (ml_pipeline/models/vectorizer.pkl)
        os.path.join(models_dir, vectorizer_filename),
        
        # Absolute path from root
        os.path.join(root_dir, models_dir, vectorizer_filename),
        
        # Within ml_pipeline directory (legacy location)
        os.path.join(current_dir, vectorizer_filename),
        os.path.join(root_dir, 'ml_pipeline', vectorizer_filename),
        
        # Within models subdirectory
        os.path.join(current_dir, 'models', vectorizer_filename),
        
        # Docker container specific paths
        f"/app/{models_dir}/{vectorizer_filename}",
        '/app/ml_pipeline/models/vectorizer.pkl',
        '/app/ml_pipeline/vectorizer.pkl',
        '/app/vectorizer.pkl',
        
        # Relative paths
        f"./{vectorizer_filename}",
        f"../{vectorizer_filename}",
        f"../ml_pipeline/{vectorizer_filename}",
        f"../ml_pipeline/models/{vectorizer_filename}",
    ]
    
    # ========================================================================
    # Search for model file
    # ========================================================================
    model_found = None
    for path in possible_model_paths:
        if path and os.path.exists(path):
            model_found = path
            print(f"Found model at: {path}")
            break
    
    if model_found is None:
        print(f"ERROR: Model not found. Tried paths: {possible_model_paths}")
        raise FileNotFoundError(
            "Model not found. Please run 'python ml_pipeline/train_model.py' first."
        )
    
    # ========================================================================
    # Search for vectorizer file
    # ========================================================================
    vectorizer_found = None
    for path in possible_vectorizer_paths:
        if path and os.path.exists(path):
            vectorizer_found = path
            print(f"Found vectorizer at: {path}")
            break
    
    if vectorizer_found is None:
        print(f"ERROR: Vectorizer not found. Tried paths: {possible_vectorizer_paths}")
        raise FileNotFoundError(
            "Vectorizer not found. Please run 'python ml_pipeline/train_model.py' first."
        )
    
    # ========================================================================
    # Load both artifacts
    # ========================================================================
    _model = joblib.load(model_found)
    _vectorizer = joblib.load(vectorizer_found)
    
    print(f"Model loaded successfully from: {model_found}")
    print(f"Vectorizer loaded successfully from: {vectorizer_found}")
    
    return _model, _vectorizer


# =============================================================================
# Text Preprocessing
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Preprocess a single text document for sentiment analysis.
    
    The preprocessing pipeline applies the following transformations:
        1. Convert to lowercase (ensures case insensitivity)
        2. Remove non-alphabetic characters (punctuation, numbers, symbols)
        3. Tokenize into individual words
        4. Remove English stopwords (common words with little semantic value)
        5. Lemmatize words to their base dictionary form
    
    Args:
        text: Raw input text (e.g., "The movie was AMAZING!!!")
    
    Returns:
        Preprocessed text string (e.g., "movie amazing")
    
    Example:
        >>> preprocess_text("The movie was AMAZING!!!")
        'movie amazing'
    """
    # Lazy load NLP resources (downloaded on first call)
    stop_words, lemmatizer = _get_nlp_resources()
    
    # Step 1: Convert to lowercase for case insensitivity
    text = text.lower()
    
    # Step 2: Remove all non-alphabetic characters (keep only a-z and spaces)
    # This removes punctuation, numbers, emojis, and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Step 3: Tokenize - split text into individual words
    words = text.split()
    
    # Step 4: Remove stopwords (common words like 'the', 'and', 'is')
    # These words don't contribute meaningfully to sentiment
    words = [w for w in words if w not in stop_words]
    
    # Step 5: Lemmatize - reduce words to their base dictionary form
    # Example: 'running' -> 'run', 'better' -> 'good', 'movies' -> 'movie'
    words = [lemmatizer.lemmatize(w) for w in words]
    
    # Step 6: Join words back into a single string
    return " ".join(words)


# =============================================================================
# Sentiment Prediction
# =============================================================================

def predict_sentiment(
    text: str,
    model: Optional[object] = None,
    vectorizer: Optional[object] = None
) -> Dict[str, Any]:
    """
    Predict sentiment of a single text review.
    
    This function is the main entry point for sentiment prediction.
    It handles text preprocessing, vectorization, model inference,
    and returns a structured result with confidence scores.
    
    Args:
        text: Raw input text (e.g., movie review like "This film was fantastic!")
        model: Pre-loaded model (optional; auto-loads if not provided)
        vectorizer: Pre-loaded vectorizer (optional; auto-loads if not provided)
    
    Returns:
        Dictionary containing:
            - sentiment: 'positive' or 'negative'
            - confidence: Probability confidence score between 0 and 1
            - raw_prediction: Binary prediction (1 for positive, 0 for negative)
            - probability_scores: Dict with probabilities for each class (if available)
    
    Example:
        >>> result = predict_sentiment("This movie was fantastic!")
        >>> print(result['sentiment'])  # 'positive'
        >>> print(result['confidence']) # 0.9876
    """
    # Load model and vectorizer if not provided
    if model is None or vectorizer is None:
        model, vectorizer = load_model_and_vectorizer()
    
    # Step 1: Preprocess the raw text (clean, remove stopwords, lemmatize)
    cleaned_text = preprocess_text(text)
    
    # Step 2: Convert text to TF-IDF vector representation
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Step 3: Get binary prediction (0 = negative, 1 = positive)
    raw_prediction = model.predict(text_vectorized)[0]
    
    # Step 4: Get probability/confidence scores if available
    if hasattr(model, 'predict_proba'):
        # For models with probability calibration (e.g., LogisticRegression)
        proba = model.predict_proba(text_vectorized)[0]
        confidence = max(proba)
        probability_scores = {
            'negative': float(proba[0]),
            'positive': float(proba[1])
        }
    else:
        # For models without predict_proba (e.g., LinearSVC)
        confidence = 1.0
        probability_scores = None
    
    # Step 5: Convert numeric prediction to human-readable sentiment
    sentiment = 'positive' if raw_prediction == 1 else 'negative'
    
    # Return structured result
    return {
        'sentiment': sentiment,
        'confidence': float(confidence),
        'raw_prediction': int(raw_prediction),
        'probability_scores': probability_scores
    }


def predict_batch(
    texts: List[str],
    model: Optional[object] = None,
    vectorizer: Optional[object] = None
) -> List[Dict[str, Any]]:
    """
    Predict sentiment for multiple texts in batch mode.
    
    This function is more efficient than calling predict_sentiment in a loop
    when processing many reviews, as it maintains the same model loading
    and resource management.
    
    Args:
        texts: List of raw input texts (e.g., multiple movie reviews)
        model: Pre-loaded model (optional; auto-loads if not provided)
        vectorizer: Pre-loaded vectorizer (optional; auto-loads if not provided)
    
    Returns:
        List of prediction dictionaries, one for each input text
    
    Example:
        >>> reviews = ["Great movie!", "Terrible film."]
        >>> results = predict_batch(reviews)
        >>> for r in results:
        ...     print(r['sentiment'])
        positive
        negative
    """
    # Load model and vectorizer once for all predictions
    if model is None or vectorizer is None:
        model, vectorizer = load_model_and_vectorizer()
    
    # Process each text and collect results
    return [predict_sentiment(text, model, vectorizer) for text in texts]


# =============================================================================
# Main Entry Point (Demo)
# =============================================================================

if __name__ == "__main__":
    """
    Demo mode: Load the model and test it with sample reviews.
    
    This allows quick testing of the prediction functionality
    without starting the API server.
    """
    print("\n" + "=" * 70)
    print("IMDB Sentiment Analysis - Prediction Demo")
    print("=" * 70)
    
    # Load model (this may take a few seconds on first run)
    print("\nLoading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer()
    print("Model ready!\n")
    
    # Test reviews covering different sentiments and complexities
    test_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Complete waste of time and money.",
        "The acting was good but the plot was boring and predictable.",
        "I don't know what to say. It was okay I guess. Nothing special.",
        "Best movie I've seen all year! Highly recommended! ★★★★★",
        "What a disaster. Poor directing, bad acting, awful script.",
        "An emotional rollercoaster from start to finish.",
        "Could have been better, but not terrible.",
    ]
    
    print("-" * 70)
    print("Testing predictions on sample reviews:")
    print("-" * 70)
    
    # Run predictions on all test reviews
    for i, review in enumerate(test_reviews, 1):
        result = predict_sentiment(review, model, vectorizer)
        
        # Truncate long reviews for display
        display_review = review[:70] + "..." if len(review) > 70 else review
        
        print(f"\n[{i}] Review: \"{display_review}\"")
        print(f"    Sentiment: {result['sentiment'].upper()}")
        print(f"    Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)