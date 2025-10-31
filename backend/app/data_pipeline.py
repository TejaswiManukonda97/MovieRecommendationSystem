import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import faiss
import numpy as np
import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import ast # To safely parse the genre string

# --- NLTK setup ---
# Download required NLTK data models
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# --- Define file paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
METADATA_CSV = os.path.join(DATA_DIR, 'movie_dataset/movies_metadata.csv') 
INDEX_FILE = os.path.join(DATA_DIR, 'movie_index.faiss')
METADATA_FILE = os.path.join(DATA_DIR, 'movies.json')
MODEL_FILE = os.path.join(DATA_DIR, 'doc2vec.model') # Path to save our new model

# --- Pre-processing setup ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Helper function to parse the 'genres' column
def parse_genres(genres_str):
    try:
        # The genres are stored as a string representation of a list of dicts
        genres_list = ast.literal_eval(genres_str)
        if isinstance(genres_list, list):
            return ' | '.join([g['name'] for g in genres_list])
    except:
        pass # Handle errors in parsing
    return "Unknown"

# --- Advanced Pre-processing Function ---
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    processed_tokens = []
    for tok in tokens:
        # Filter out stopwords and punctuation
        if tok not in stop_words and tok not in punctuation:
            # Lemmatize the token (e.g., 'running' -> 'run')
            lem_tok = lemmatizer.lemmatize(tok)
            processed_tokens.append(lem_tok)
    return processed_tokens

def build_and_save_index():
    print("Starting data pipeline with NEW (movies_metadata.csv) dataset...")
    
    # Load the new dataset
    try:
        # Specify dtype for 'id' to avoid mixing types
        movies = pd.read_csv(METADATA_CSV, dtype={'id': str})
    except Exception as e:
        print(f"Error reading {METADATA_CSV}. Make sure it's in the backend/data/ folder.")
        print(f"Error: {e}")
        return

    # --- 1. Clean and Prepare Data ---
    print("Cleaning and preparing data...")
    # Drop movies with no overview or title
    movies = movies.dropna(subset=['overview', 'title'])
    # Reset index to ensure 0-based integer indexing for tags
    movies = movies.reset_index(drop=True)

    # --- 2. Create Rich Text Representation ---
    def create_text_rep(row):
        title = row['title']
        genres = parse_genres(row['genres'])
        overview = row['overview']
        # Combine them into one "document"
        return f"{title}. Genres: {genres}. Plot: {overview}"

    movies['text_representation'] = movies.apply(create_text_rep, axis=1)

    # --- 3. Create TaggedDocument for Doc2Vec (Now uses pre-processing) ---
    print("Preparing data for Doc2Vec with ADVANCED pre-processing...")
    tagged_data = []
    metadata = {} # We'll build metadata here
    
    for idx, row in movies.iterrows():
        idx = int(idx) # Ensure native int for keys
        
        # Use our new pre-processing function
        processed_tokens = preprocess_text(row['text_representation'])
        
        tagged_data.append(TaggedDocument(words=processed_tokens, tags=[str(idx)]))
        metadata[idx] = row['title']

    # --- 4. Train Fine-Tuned Doc2Vec Model ( Hyperparameters) ---
    print(f"Training FINE-TUNED Doc2Vec model on {len(tagged_data)} documents...")
    
    model = Doc2Vec(tagged_data,
                    vector_size=300,  # Increased from 100
                    window=8,         # Increased from 5
                    min_count=5,      # Increased from 2 (ignores very rare words)
                    epochs=80,        # Increased from 40 (more training passes)
                    workers=4)        # Use 4 CPU cores
    
    print("Training complete. Saving model...")
    model.save(MODEL_FILE)

    # --- 5. Build and save HNSW index ( Hyperparameters) ---
    print("Building FAISS index from Doc2Vec vectors...")
    dimension = model.vector_size
    
    embeddings = np.array([model.dv[str(i)] for i in range(model.corpus_count)]).astype('float32')

    # 1. Increase M from 32 to 128 for a denser graph
    m = 128 
    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
    
    # 2. Set efConstruction before adding data (default is 40)
    # We increase it for a higher quality index
    index.hnsw.efConstruction = 256 
    
    print("FAISS index parameters set. Starting to add vectors...")
    
    faiss.normalize_L2(embeddings)
    
    # This.add() step will now be slower, but the result is better
    index.add(embeddings)
    
    print(f"Index built with {index.ntotal} vectors.")
    faiss.write_index(index, INDEX_FILE)
    
    # --- 6. Save metadata ---
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

    print(f"Pipeline complete. All fine-tuned model files saved.")

if __name__ == "__main__":
    build_and_save_index()