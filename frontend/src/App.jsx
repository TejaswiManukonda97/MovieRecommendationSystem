import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [query, setQuery] = useState('')
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const API_URL = "http://localhost:8000/recommend"

  const handleSearch = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setRecommendations([])

    try {
      const response = await axios.post(API_URL, {
        query: query,
        k: 10
      })
      setRecommendations(response.data.recommendations)
    } catch (err) {
      setError("Failed to fetch recommendations. Is the backend server running?")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <header>
        <h1>AI Movie Recommender</h1>
        <p>Built with React, FastAPI, and FAISS (HNSW)</p>
      </header>

      <form onSubmit={handleSearch} className="search-form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter a movie title or theme (e.g., 'Toy Story' or 'space war')"
        />
        <button type="submit" disabled={loading}>
          {loading? 'Searching...' : 'Recommend'}
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}

      <div className="results">
        {recommendations.length > 0 && <h2>Recommendations:</h2>}
        <ul className="recommendation-list">
          {recommendations.map((movie) => (
            <li key={movie.title} className="movie-item">
              <h3>{movie.title}</h3>
              <p>Similarity: {(movie.score * 100).toFixed(1)}%</p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export default App