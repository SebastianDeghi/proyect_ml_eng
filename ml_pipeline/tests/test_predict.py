"""
test_predict.py - Unit tests for prediction module

Usage:
    pytest tests/test_predict.py -v
    pytest tests/test_predict.py -v --cov=predict
"""

import os
import sys
import pytest
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import (
    preprocess_text,
    predict_sentiment,
    load_model_and_vectorizer
)


# =============================================
# Fixtures
# =============================================

@pytest.fixture(scope="session")
def model_and_vectorizer():
    """Load model and vectorizer once for all tests."""
    model, vectorizer = load_model_and_vectorizer()
    return model, vectorizer


@pytest.fixture
def sample_positive_reviews():
    """Sample positive reviews for testing."""
    return [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Great film, fantastic acting, wonderful story.",
        "Best movie I've seen all year. Highly recommended!",
        "Excellent performance by the entire cast.",
        "A masterpiece of modern cinema."
    ]


@pytest.fixture
def sample_negative_reviews():
    """Sample negative reviews for testing."""
    return [
        "Terrible film. Waste of time and money.",
        "Boring, predictable, and poorly acted.",
        "Worst movie of the decade. Awful in every way.",
        "Don't waste your time on this garbage.",
        "Nothing good about this film. Complete disaster."
    ]


@pytest.fixture
def sample_neutral_reviews():
    """Sample neutral/ambiguous reviews for testing."""
    return [
        "It was okay, nothing special.",
        "The acting was good but the plot was boring.",
        "I don't know what to say. It was fine I guess.",
        "Decent movie but not great.",
        "Average film, some good parts some bad."
    ]


# =============================================
# Tests for preprocess_text
# =============================================

class TestPreprocessText:
    """Test text preprocessing functionality."""
    
    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase."""
        result = preprocess_text("HELLO WORLD")
        assert result == "hello world"
    
    def test_remove_punctuation(self):
        """Test that punctuation is removed."""
        result = preprocess_text("Hello, world!!! How are you?")
        assert "!" not in result
        assert "?" not in result
        assert "," not in result
    
    def test_remove_stopwords(self):
        """Test that stopwords are removed."""
        result = preprocess_text("the and a an this is that movie")
        # Stopwords should be removed, 'movie' should remain
        assert "movie" in result
        assert "the" not in result
        assert "and" not in result
    
    def test_lemmatization(self):
        """Test that words are lemmatized."""
        result = preprocess_text("running better movies")
        assert "run" in result      # 'running' → 'run'
        assert "movie" in result    # 'movies' → 'movie'
        # 'better' is an irregular comparative; NLTK keeps it as 'better'
        # This is expected behavior

    
    def test_empty_text(self):
        """Test empty text input."""
        result = preprocess_text("")
        assert result == ""
    
    def test_only_stopwords(self):
        """Test text with only stopwords."""
        result = preprocess_text("the and a an")
        assert result == ""


# =============================================
# Tests for predict_sentiment
# =============================================

class TestPredictSentiment:
    """Test sentiment prediction functionality."""
    
    def test_positive_prediction(self, model_and_vectorizer, sample_positive_reviews):
        """Test that positive reviews are predicted as positive."""
        model, vectorizer = model_and_vectorizer
        
        for review in sample_positive_reviews:
            result = predict_sentiment(review, model, vectorizer)
            assert result['sentiment'] == 'positive'
            assert result['confidence'] >= 0.5
            assert result['raw_prediction'] == 1
    
    def test_negative_prediction(self, model_and_vectorizer, sample_negative_reviews):
        """Test that negative reviews are predicted as negative."""
        model, vectorizer = model_and_vectorizer
        
        for review in sample_negative_reviews:
            result = predict_sentiment(review, model, vectorizer)
            assert result['sentiment'] == 'negative'
            assert result['confidence'] >= 0.5
            assert result['raw_prediction'] == 0
    
    def test_confidence_range(self, model_and_vectorizer):
        """Test that confidence is between 0 and 1."""
        model, vectorizer = model_and_vectorizer
        
        test_text = "This movie is good."
        result = predict_sentiment(test_text, model, vectorizer)
        
        assert 0 <= result['confidence'] <= 1
    
    def test_return_structure(self, model_and_vectorizer):
        """Test that return dictionary has expected keys."""
        model, vectorizer = model_and_vectorizer
        
        result = predict_sentiment("Good movie", model, vectorizer)
        
        expected_keys = {'sentiment', 'confidence', 'raw_prediction', 'probability_scores'}
        assert expected_keys.issubset(result.keys())
    
    def test_empty_string_handling(self, model_and_vectorizer):
        """Test that empty string is handled gracefully."""
        model, vectorizer = model_and_vectorizer
        
        # Empty string should still return a prediction
        result = predict_sentiment("", model, vectorizer)
        assert result['sentiment'] in ['positive', 'negative']
        assert isinstance(result['confidence'], float)


# =============================================
# Tests for batch prediction (via predict_batch)
# =============================================

class TestBatchPrediction:
    """Test batch prediction functionality."""
    
    def test_batch_prediction(self, model_and_vectorizer):
        """Test predicting multiple texts at once."""
        from predict import predict_batch
        
        model, vectorizer = model_and_vectorizer
        
        texts = [
            "Great movie!",
            "Terrible film.",
            "It was okay."
        ]
        
        results = predict_batch(texts, model, vectorizer)
        
        assert len(results) == 3
        for result in results:
            assert 'sentiment' in result
            assert 'confidence' in result
    
    def test_batch_empty_list(self, model_and_vectorizer):
        """Test batch prediction with empty list."""
        from predict import predict_batch
        
        model, vectorizer = model_and_vectorizer
        
        results = predict_batch([], model, vectorizer)
        assert results == []


# =============================================
# Tests for model loading
# =============================================

class TestModelLoading:
    """Test model loading functionality."""
    
    def test_model_exists(self):
        """Test that model file exists in the models directory."""
        import yaml
        
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        models_dir = config['paths']['models_dir']
        model_path = os.path.join(models_dir, config['paths']['model_filename'])
        assert os.path.exists(model_path), f"Model not found at {model_path}"
    
    def test_vectorizer_exists(self):
        """Test that vectorizer file exists in the models directory."""
        import yaml
        
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        models_dir = config['paths']['models_dir']
        vectorizer_path = os.path.join(models_dir, config['paths']['vectorizer_filename'])
        assert os.path.exists(vectorizer_path), f"Vectorizer not found at {vectorizer_path}"
    
    def test_model_loads_successfully(self):
        """Test that model loads without errors."""
        model, vectorizer = load_model_and_vectorizer()
        assert model is not None
        assert vectorizer is not None
    
    def test_model_can_predict(self, model_and_vectorizer):
        """Test that loaded model can make predictions."""
        model, vectorizer = model_and_vectorizer
        
        # Should not raise exception
        result = predict_sentiment("Test movie", model, vectorizer)
        assert isinstance(result, dict)


# =============================================
# Run tests if executed directly
# =============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])