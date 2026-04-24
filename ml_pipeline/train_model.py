"""
train_model.py - Script to train the best sentiment analysis model

This script:
1. Downloads/loads the IMDB dataset
2. Preprocesses text (cleaning, stopwords removal, lemmatization)
3. Trains TF-IDF vectorizer
4. Trains Logistic Regression (best model)
5. Saves model and vectorizer to disk
6. Optionally trains Word2Vec for reference

Usage:
    python train_model.py
"""

import os
import re
import sys
import yaml
import joblib
import pandas as pd
import numpy as np
import kagglehub
import shutil
import nltk
from collections import Counter

# NLTK downloads
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional: Word2Vec
try:
    from gensim.models import Word2Vec
    W2V_AVAILABLE = True
except ImportError:
    W2V_AVAILABLE = False
    print("Warning: gensim not installed. Word2Vec training will be skipped.")


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    # Try multiple locations
    possible_paths = [
        config_path,
        os.path.join(os.path.dirname(__file__), config_path),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path),
        os.path.join(os.getcwd(), config_path),
        '/app/config.yaml',  # Docker path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return config
    
    raise FileNotFoundError(f"Config file not found. Tried: {possible_paths}")


def download_and_load_dataset(config):
    """Download dataset from Kaggle and load into DataFrame."""
    print("=" * 60)
    print("STEP 1: Downloading and loading dataset")
    print("=" * 60)
    
    # Download from Kaggle
    dataset_path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Find CSV file
    csv_file = None
    for f in os.listdir(dataset_path):
        if f.endswith(".csv"):
            csv_file = os.path.join(dataset_path, f)
            break
    
    if csv_file is None:
        raise FileNotFoundError("No CSV file found in downloaded dataset")
    
    # Create local folder
    local_folder = config['data']['local_folder']
    os.makedirs(local_folder, exist_ok=True)
    
    # Copy to local folder
    local_csv = os.path.join(local_folder, config['data']['csv_filename'])
    shutil.copy(csv_file, local_csv)
    print(f"Dataset copied to: {local_csv}")
    
    # Load DataFrame
    df = pd.read_csv(local_csv)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Class distribution:\n{df['sentiment'].value_counts()}")
    
    return df


def preprocess_text(text, stop_words, lemmatizer):
    """
    Preprocess a single text document.
    
    Steps:
    1. Lowercase
    2. Remove non-alphabetic characters
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    """
    # Lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    words = [w for w in words if w not in stop_words]
    
    # Lemmatize
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)


def preprocess_dataset(df, config):
    """Preprocess all reviews in the dataset."""
    print("\n" + "=" * 60)
    print("STEP 2: Text preprocessing")
    print("=" * 60)
    
    # Initialize stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    print("Preprocessing 50,000 reviews... (this may take 2-3 minutes)")
    
    # Apply preprocessing
    df['clean_review'] = df['review'].apply(
        lambda x: preprocess_text(x, stop_words, lemmatizer)
    )
    
    # Encode labels
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    print("Preprocessing completed")
    print(f"Sample cleaned review: {df['clean_review'].iloc[0][:200]}...")
    
    return df


def train_tfidf_model(df, config):
    """Train TF-IDF vectorizer and Logistic Regression model."""
    print("\n" + "=" * 60)
    print("STEP 3: Training TF-IDF + Logistic Regression")
    print("=" * 60)
    
    # Split data
    X = df['clean_review']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=config['preprocessing']['max_features'],
        ngram_range=tuple(config['preprocessing']['ngram_range'])
    )
    
    # Fit and transform
    print("Vectorizing training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    
    # Train Logistic Regression
    model_params = config['best_model']['parameters']
    model = LogisticRegression(
        C=model_params['C'],
        solver=model_params['solver'],
        max_iter=model_params['max_iter'],
        random_state=model_params['random_state']
    )
    
    print("Training Logistic Regression...")
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print("\nModel Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    
    return model, vectorizer, metrics


def train_word2vec(df, config):
    """Optional: Train Word2Vec embeddings."""
    if not W2V_AVAILABLE:
        print("\nSkipping Word2Vec training (gensim not installed)")
        return None
    
    print("\n" + "=" * 60)
    print("STEP 4 (Optional): Training Word2Vec embeddings")
    print("=" * 60)
    
    # Split data
    X = df['clean_review']
    y = df['label']
    
    X_train, _, _, _ = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    # Tokenize
    tokenized_train = [txt.split() for txt in X_train]
    
    # Train Word2Vec
    w2v_config = config['word2vec']
    w2v_model = Word2Vec(
        sentences=tokenized_train,
        vector_size=w2v_config['vector_size'],
        window=w2v_config['window'],
        min_count=w2v_config['min_count'],
        sg=w2v_config['sg'],
        workers=w2v_config['workers'],
        seed=w2v_config['seed'],
        epochs=w2v_config['epochs']
    )
    
    print(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
    print(f"Vector dimension: {w2v_model.wv.vector_size}")
    
    # Test similarity
    if 'good' in w2v_model.wv:
        print("\nSemantic neighbors of 'good':")
        for word, sim in w2v_model.wv.most_similar('good', topn=5):
            print(f"  {word}: {sim:.3f}")
    
    return w2v_model


def save_artifacts(model, vectorizer, w2v_model, config):
    """Save model and vectorizer to disk."""
    print("\n" + "=" * 60)
    print("STEP 5: Saving artifacts")
    print("=" * 60)
    
    # Get models directory from config
    models_dir = config['paths']['models_dir']
    
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models directory: {os.path.abspath(models_dir)}")
    
    # Save model
    model_path = os.path.join(models_dir, config['paths']['model_filename'])
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(models_dir, config['paths']['vectorizer_filename'])
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Save Word2Vec if available
    if w2v_model is not None:
        w2v_path = os.path.join(models_dir, config['paths']['w2v_model_filename'])
        w2v_model.save(w2v_path)
        print(f"Word2Vec model saved to: {w2v_path}")
    
    print("\nAll artifacts saved successfully!")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("IMDB Sentiment Analysis - Training Pipeline")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Run pipeline
    df = download_and_load_dataset(config)
    df = preprocess_dataset(df, config)
    model, vectorizer, metrics = train_tfidf_model(df, config)
    w2v_model = train_word2vec(df, config)
    save_artifacts(model, vectorizer, w2v_model, config)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Best F1-score: {metrics['f1']:.4f}")
    print("=" * 60)
    
    return model, vectorizer, metrics


if __name__ == "__main__":
    main()