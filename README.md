# MovieRecommendationSystem
Recommend 10 movies that are similar to the input based on plot, genre and ratings

This repository contains the foundational structure for the Recommendation System Inference API built with FastAPI and containerized with Docker, ready for deployment in a CI/CD pipeline.

Local Setup (macOS)

Prerequisites

You must have the following installed on your Mac:

Python 3.10+

pip (Python package installer)

virtualenv or similar tool (optional, but highly recommended)

Docker Desktop

1. Set up Python Environment (Recommended)

Create and activate a virtual environment to manage dependencies cleanly.

# Create a virtual environment
python3 -m venv .venv
# Activate the environment
source .venv/bin/activate


2. Install Dependencies

Install all required Python packages listed in requirements.txt:

pip install -r requirements.txt


3. Run Locally (without Docker)

For rapid development with hot-reloading:

# The 'main' is the file (app/main.py, but Python finds it as main when in a venv/app context)
# The 'app' is the FastAPI instance (app = FastAPI())
# The '--reload' flag enables automatic server restart on file changes
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


The API will be running at http://127.0.0.1:8000. Open http://127.0.0.1:8000/docs to see the interactive API documentation (Swagger UI).

🐳 Docker Setup

Containerizing the application ensures your local environment exactly matches the future cloud environment, which is crucial for MLOps.

1. Build the Docker Image

Run this command from the root directory (mlops-recommendation-api/):

docker build -t recommendation-api:local .


2. Run the Docker Container

Run the image, mapping the container's internal port 8000 to your Mac's local port 8000:

docker run -d -p 8000:8000 --name recommendation-api-container recommendation-api:local


3. Test the API

Access the documentation at: http://localhost:8000/docs

You can test the /predict endpoint using curl:

curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"user_id": "test_user_001", "feature_vector": [1.0, 1.0]}'
