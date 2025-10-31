import faiss
import json
import os
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# --- NLTK setup ---
# Download required NLTK data models
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Define file paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
INDEX_FILE = os.path.join(DATA_DIR, 'movie_index.faiss')
METADATA_FILE = os.path.join(DATA_DIR, 'movies.json')
MODEL_FILE = os.path.join(DATA_DIR, 'doc2vec.model') # Path to our saved model

class Recommender:
    def __init__(self):
        print("Loading recommender model...")
        self.index = faiss.read_index(INDEX_FILE)
        
        # --- Tuned HNSW Search Parameter ---
        # Set the search-time "thoroughness"
        # Higher = more accurate, Slower = faster
        self.index.hnsw.efSearch = 128
        
        with open(METADATA_FILE, 'r') as f:
            # Load metadata and convert keys back to integers
            self.metadata = {int(k): v for k, v in json.load(f).items()}
            
        self.model = Doc2Vec.load(MODEL_FILE)
        
        # --- Setup pre-processing tools for the class ---
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        
        print("Recommender model loaded.")

    # --- New Advanced Pre-processing Function (for queries) ---
    def _preprocess_query(self, text):
        tokens = word_tokenize(text.lower())
        processed_tokens = []
        for tok in tokens:
            if tok not in self.stop_words and tok not in self.punctuation:
                lem_tok = self.lemmatizer.lemmatize(tok)
                processed_tokens.append(lem_tok)
        return processed_tokens

    def get_recommendations(self, query: str, k: int = 10):
        # 1. Tokenize and infer vector for the query (using new method)
        processed_tokens = self._preprocess_query(query)
        
        query_vector = self.model.infer_vector(processed_tokens)
        
        query_vector = np.array([query_vector]).astype('float32') # Needs to be 2D for FAISS
        faiss.normalize_L2(query_vector)

        # 2. Search the index
        # D (distances) and I (indices) are 2D arrays
        D, I = self.index.search(query_vector, k)

        # 3. Map indices back to movie titles
        results = []
        
        # Get the first (and only) row of results
        indices_row = I[0]
        distances_row = D[0]

        # Now, iterate over the 1D arrays
        for i in range(len(indices_row)):
            # Get the integer index from the 1D array
            idx = int(indices_row[i])
            
            if idx in self.metadata:
                results.append({
                    "title": self.metadata[idx],
                    "score": float(distances_row[i]) # Get the corresponding score
                })

        return results