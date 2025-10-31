from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from.model import Recommender

app = FastAPI(title="Movie Recommender API")

# --- CORS Middleware ---
# This allows your React frontend (running on a different port)
# to communicate with this backend.
origins = origins = [
    "http://localhost:5173",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model ---
# This loads the model once when the server starts
recommender = Recommender()

# --- API Data Models ---
class RecommendQuery(BaseModel):
    query: str
    k: int = 10

class Movie(BaseModel):
    title: str
    score: float

class RecommendResponse(BaseModel):
    recommendations: list[Movie]

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Recommender API"}

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(query: RecommendQuery):
    """
    Get movie recommendations based on a text query.
    """
    recommendations = recommender.get_recommendations(query.query, query.k)
    return {"recommendations": recommendations}