# 🎬 IMDB Sentiment Analysis — From NLP to Production

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLOps](https://img.shields.io/badge/MLOps-Ready-blue)](https://mlops.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-17%20passed-brightgreen)](https://github.com/SebastianDeghi/proyect_ml_eng/actions)
![Views](https://komarev.com/ghpvc/?username=SebastianDeghi&color=blue&style=flat)

### 👥 Contributors: **Emmanuel Gonzalez Gomez** & **Dalma Márquez**

---

## 📝 Overview

This project performs **binary sentiment classification** (positive/negative) on 50,000 IMDB movie reviews.

It combines:

- Traditional NLP techniques (**TF-IDF**)
- Dense semantic representations (**Word2Vec embeddings**)
- Multiple machine learning models
- **Production deployment using FastAPI**

The project evolves from a Data Science workflow into a **Machine Learning Engineering solution**, where the best model is deployed as a REST API with Docker containerization and CI/CD pipelines.

---

## 📁 Project Structure

```
proyect_ml_eng/
├── 📂 .github/workflows/
│ └── ci_cd.yml                      # CI/CD pipeline with GitHub Actions
├── 📂 ml_pipeline/
│ ├── init.py
│ ├── train_model.py                 # Model training script
│ ├── predict.py                     # Prediction module
│ ├── config.yaml                    # Configuration file
│ ├── 📂 models/
│ │ ├── model.pkl                    # Saved models (generated)
│ │ └── vectorizer.pkl
│ ├── 📂 api/
│ │ ├── init.py
│ │ ├── app.py                      # FastAPI application
│ │ └── Dockerfile                  # Containerization
│ └── 📂 tests/
│ ├── init.py
│ └── test_predict.py               # Unit tests (17 tests)
├── 📂 notebooks/
│ └── IMDB_NLP_Sentiment_Analysis.ipynb # Original EDA and modeling
├── 📂 examples/
│ ├── api_examples.py               # API usage examples
│ └── notebook_usage.ipynb          # Notebook usage example
├── 📂 scripts/
│ ├── download_dataset.py           # Dataset download utility
│ └── benchmark_model.py            # Performance benchmarks
├── 📂 monitoring/
│ └── prometheus.yml                # Prometheus configuration
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── docker-compose.yml              # Multi-container setup
├── Makefile                        # Common commands
├── pyproject.toml                  # Project configuration
├── .pre-commit-config.yaml         # Pre-commit hooks
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

The **IMDB Dataset of 50K Movie Reviews** contains labeled movie reviews.

- **Size:** 50,000 reviews  
- **Distribution:** Balanced (25k positive / 25k negative)  
- **Features:**
  - `review`: raw text
  - `sentiment`: label  
- **Task:** Binary classification  

**Source:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## 🚀 Models and Representations

### Text Representations

| Representation | Type | Description |
| :--- | :--- | :--- |
| **TF-IDF** | Sparse | Statistical weighting based on term frequency |
| **Word2Vec** | Dense | Semantic embeddings trained with Skip-gram |

### Models Evaluated

| Representation | Model | Accuracy | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF** | Logistic Regression | 0.8944 | 0.8865 | 0.9046 | **0.8955** |
| | LinearSVC | 0.8920 | 0.8940 | 0.8930 | 0.8937 |
| | Naive Bayes | 0.8670 | 0.8700 | 0.8680 | 0.8688 |
| **Word2Vec** | MLP | 0.8810 | 0.8830 | 0.8820 | **0.8825** |
| | SVC (RBF) | 0.8800 | 0.8820 | 0.8810 | 0.8810 |
| | Logistic Regression | 0.8710 | 0.8730 | 0.8720 | 0.8720 |
| | Random Forest | 0.8640 | 0.8660 | 0.8650 | 0.8650 |

---

## 🧠 Key Insights

- TF-IDF + Logistic Regression provides the best performance (F1: 0.8955)
- Linear models work best with sparse representations
- Non-linear models benefit from dense embeddings
- Word2Vec captures semantic relationships better but is slightly less accurate here

---

## ⚠️ Known Limitations

- Does not handle neutral sentiment (binary only)
- TF-IDF ignores word order and context
- Vocabulary limited to training set (new words become zero vectors)
- Inference time: ~15ms per request on CPU

---

## ⚙️ Production API (FastAPI)

The best model (**TF-IDF + Logistic Regression**) is deployed as a REST API with the following endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | GET | Health check |
| `/predict` | POST | Single sentiment prediction |
| `/batch` | POST | Batch sentiment prediction |
| `/docs` | GET | Interactive API documentation |
| `/info` | GET | Model information and metrics |

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip
- (Optional) Docker Desktop

### Clone the repository

```bash
git clone https://github.com/SebastianDeghi/proyect_ml_eng.git
cd proyect_ml_eng
```

### Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the model

```bash
python ml_pipeline/train_model.py
```

This will:
- Download the IMDB dataset
- Preprocess 50,000 reviews
- Train TF-IDF vectorizer and Logistic Regression
- Save artifacts to `ml_pipeline/models/`

### 2. Run tests

```bash
pytest ml_pipeline/tests/ -v
```

Expected output: `17 passed`

### 3. Test predictions

```bash
python -c "from ml_pipeline.predict import predict_sentiment, load_model_and_vectorizer; m, v = load_model_and_vectorizer(); print('Positive:', predict_sentiment('Great movie!', m, v)['sentiment']); print('Negative:', predict_sentiment('Awful film', m, v)['sentiment'])"
```

### 4. Run API locally

```bash
cd ml_pipeline/api
uvicorn app:app --reload
```

Open in your browser:

👉 http://127.0.0.1:8000/docs


### 5. Test the API (with PowerShell)

```bash
# Positive review
$response = Invoke-WebRequest -Uri http://localhost:8000/predict -Method POST -Body '{"text":"This movie is absolutely amazing!"}' -ContentType "application/json"
$response.Content | ConvertFrom-Json

# Negative review
$response = Invoke-WebRequest -Uri http://localhost:8000/predict -Method POST -Body '{"text":"Terrible film, waste of time."}' -ContentType "application/json"
$response.Content | ConvertFrom-Json
```

### 6. Run API examples

```bash
python examples/api_examples.py
```

## 🐳 Docker

### Build the image

```bash
docker build -t imdb-api -f ml_pipeline/api/Dockerfile .
```

### Run the container

```bash
docker run -p 8000:8000 imdb-api
```

### Using Docker Compose

```bash
docker-compose up --build
```

---

## 🔮 API Endpoint

**POST** `/predict`

### Request

```json
{
  "text": "This movie was absolutely amazing!"
}
```

### Response

```json
{
  "sentiment": "positive",
  "confidence": 0.9876,
  "text_length": 42
}
```

**POST** `/batch`

### Request

```json
{
  "texts": [
    "Great movie!",
    "Awful film.",
    "It was okay."
  ]
}
```

### Response

```json
{
  "results": [
    {"sentiment": "positive", "confidence": 0.95, "text_length": 12},
    {"sentiment": "negative", "confidence": 0.92, "text_length": 11},
    {"sentiment": "negative", "confidence": 0.65, "text_length": 13}
  ],
  "total_count": 3
}
```

**GET** `/health`

### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## 🧪 Inference Pipeline

```
Raw Text → Preprocessing → TF-IDF Vectorization → Logistic Regression → Sentiment Prediction
```

Preprocessing steps:
1. Lowercase conversion
2. Remove non-alphabetic characters
3. Tokenization
4. Stopwords removal
5. Lemmatization

---

## 🗺️ Roadmap

- TF-IDF + Logistic Regression baseline
- FastAPI deployment
- Docker containerization
- Unit tests (17 tests passing)
- Batch prediction endpoint
- MLflow experiment tracking
- Prometheus monitoring
- GitHub Actions CI/CD
- Cloud deployment (Render/AWS)

---

## 📊 Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 89.44% |
| **Precision** | 88.65% |
| **Recall** | 90.46% |
| **F1-score** | 89.55% |
| **Inference Time** | ~15ms per request |
| **Training Time** | ~45 seconds |

---

## 📚 References

- https://aclanthology.org/P11-1015.pdf  
- https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews  

---

## 🙏 Acknowledgements

- **Lakshmi N Pathi** for making the dataset publicly available on Kaggle  
- The open-source community for essential tools:
  - `scikit-learn`
  - `nltk`
  - `gensim`
  - `pandas`
  - `numpy`
  - `fastapi`
  - `uvicorn`

---

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sebastián Deghi**

- GitHub: https://github.com/SebastianDeghi  
- LinkedIn: https://www.linkedin.com/in/sebastian-deghi/  
- Google Scholar: https://scholar.google.com/citations?user=3Nq5hTIAAAAJ&hl=en  