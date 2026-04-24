# =============================================
# MAKEFILE - IMDB Sentiment Analysis
# =============================================
#
# Commands:
#   make help          - Show all commands
#   make setup         - Setup virtual environment and install dependencies
#   make train         - Train the model
#   make test          - Run tests
#   make api           - Run API locally
#   make docker-build  - Build Docker image
#   make docker-run    - Run Docker container
#   make docker-compose-up - Start all services with docker-compose
#   make docker-compose-down - Stop all services
#   make clean         - Clean temporary files
#   make lint          - Run code linters
#   make format        - Format code with black
#   make all           - Run full pipeline (setup → train → test)
#
# =============================================

.PHONY: help setup train test api docker-build docker-run docker-compose-up docker-compose-down clean lint format all

# Variables
PYTHON = python3
VENV = venv
VENV_ACTIVATE = . $(VENV)/bin/activate
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
PYTHON_VENV = $(VENV)/bin/python

# Colors for output
GREEN = \033[0;32m
RED = \033[0;31m
NC = \033[0m # No Color

help:
	@echo "$(GREEN)IMDB Sentiment Analysis - Makefile Commands$(NC)"
	@echo ""
	@echo "  $(GREEN)make setup$(NC)         - Setup virtual environment and install dependencies"
	@echo "  $(GREEN)make train$(NC)         - Train the sentiment analysis model"
	@echo "  $(GREEN)make test$(NC)          - Run unit tests"
	@echo "  $(GREEN)make api$(NC)           - Run FastAPI locally"
	@echo "  $(GREEN)make docker-build$(NC)  - Build Docker image"
	@echo "  $(GREEN)make docker-run$(NC)    - Run Docker container"
	@echo "  $(GREEN)make docker-compose-up$(NC)   - Start all services with docker-compose"
	@echo "  $(GREEN)make docker-compose-down$(NC) - Stop all services"
	@echo "  $(GREEN)make clean$(NC)         - Clean temporary files"
	@echo "  $(GREEN)make lint$(NC)          - Run code linters"
	@echo "  $(GREEN)make format$(NC)        - Format code with black"
	@echo "  $(GREEN)make all$(NC)           - Run full pipeline (setup → train → test)"
	@echo ""

setup:
	@echo "$(GREEN)Setting up virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)Downloading NLTK data...$(NC)"
	@$(PYTHON_VENV) -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
	@echo "$(GREEN)Setup completed!$(NC)"
	@echo "$(GREEN)Activate environment with: source $(VENV)/bin/activate$(NC)"

train:
	@echo "$(GREEN)Training model...$(NC)"
	@$(PYTHON_VENV) ml_pipeline/train_model.py

test:
	@echo "$(GREEN)Running tests...$(NC)"
	@$(PYTEST) ml_pipeline/tests/ -v --tb=short

api:
	@echo "$(GREEN)Starting FastAPI server...$(NC)"
	@cd ml_pipeline/api && $(PYTHON_VENV) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	@echo "$(GREEN)Building Docker image...$(NC)"
	@docker build -t imdb-sentiment-api -f ml_pipeline/api/Dockerfile .

docker-run:
	@echo "$(GREEN)Running Docker container...$(NC)"
	@docker run -p 8000:8000 --name imdb-sentiment-api imdb-sentiment-api

docker-compose-up:
	@echo "$(GREEN)Starting services with docker-compose...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)Services started:$(NC)"
	@echo "  API: http://localhost:8000"
	@echo "  Docs: http://localhost:8000/docs"
	@echo "  Prometheus: http://localhost:9090 (with --profile monitoring)"
	@echo "  Grafana: http://localhost:3000 (with --profile monitoring)"

docker-compose-down:
	@echo "$(GREEN)Stopping services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)Services stopped$(NC)"

docker-compose-monitoring:
	@echo "$(GREEN)Starting services with monitoring...$(NC)"
	@docker-compose --profile monitoring up -d
	@echo "$(GREEN)Monitoring services started:$(NC)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"

clean:
	@echo "$(GREEN)Cleaning temporary files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".DS_Store" -delete
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf *.egg-info/
	@rm -rf dist/
	@rm -rf build/
	@echo "$(GREEN)Clean completed!$(NC)"

clean-models:
	@echo "$(GREEN)Removing trained models...$(NC)"
	@rm -rf ml_pipeline/models/*.pkl
	@rm -rf ml_pipeline/models/*.model
	@echo "$(GREEN)Models cleaned!$(NC)"

clean-all: clean clean-models
	@echo "$(GREEN)Removing virtual environment and datasets...$(NC)"
	@rm -rf $(VENV)
	@rm -rf imdb_dataset/
	@echo "$(GREEN)All cleaned!$(NC)"

lint:
	@echo "$(GREEN)Running linters...$(NC)"
	@$(PIP) install flake8 black 2>/dev/null || true
	@$(VENV)/bin/flake8 ml_pipeline/ --max-line-length=120 --ignore=E203,W503
	@echo "$(GREEN)Lint completed!$(NC)"

format:
	@echo "$(GREEN)Formatting code...$(NC)"
	@$(PIP) install black 2>/dev/null || true
	@$(VENV)/bin/black ml_pipeline/
	@echo "$(GREEN)Format completed!$(NC)"

all: setup train test
	@echo "$(GREEN)Full pipeline completed!$(NC)"