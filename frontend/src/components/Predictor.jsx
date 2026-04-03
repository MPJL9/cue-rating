import React, { useState, useEffect, useCallback } from 'react'
import { searchPlayers, predictMatch } from '../api'

function PlayerSearch({ label, value, onChange }) {
  const [query, setQuery] = useState(value)
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (query.length < 2) { setResults([]); return }
    const timer = setTimeout(() => {
      searchPlayers(query)
        .then(data => setResults(data.results.slice(0, 8)))
        .catch(() => setResults([]))
    }, 300)
    return () => clearTimeout(timer)
  }, [query])

  return (
    <label>
      {label}
      <div style={{ position: 'relative' }}>
        <input
          value={query}
          onChange={e => { setQuery(e.target.value); setOpen(true) }}
          onFocus={() => setOpen(true)}
          placeholder="Type player name..."
        />
        {open && results.length > 0 && (
          <div style={{
            position: 'absolute', top: '100%', left: 0, right: 0,
            background: '#1a1a1a', border: '1px solid #333', borderRadius: 6,
            zIndex: 10, maxHeight: 200, overflow: 'auto',
          }}>
            {results.map(r => (
              <div
                key={r.name}
                onClick={() => { setQuery(r.name); onChange(r.name); setOpen(false) }}
                style={{
                  padding: '0.4rem 0.75rem', cursor: 'pointer',
                  borderBottom: '1px solid #222', fontSize: '0.9rem',
                }}
                onMouseEnter={e => e.target.style.background = '#222'}
                onMouseLeave={e => e.target.style.background = 'transparent'}
              >
                {r.name} <span style={{ color: '#666' }}>({r.elo_rating})</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </label>
  )
}

export default function Predictor() {
  const [player1, setPlayer1] = useState('')
  const [player2, setPlayer2] = useState('')
  const [bestOf, setBestOf] = useState(9)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)

  const handlePredict = useCallback(() => {
    if (!player1 || !player2) return
    setLoading(true)
    predictMatch(player1, player2, bestOf)
      .then(setPrediction)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [player1, player2, bestOf])

  return (
    <>
      <div className="card">
        <h2>Match Predictor</h2>
        <div className="predict-form">
          <PlayerSearch label="Player 1" value={player1} onChange={setPlayer1} />
          <span style={{ color: '#666', fontWeight: 700, fontSize: '1.2rem', alignSelf: 'center', paddingTop: '1.2rem' }}>vs</span>
          <PlayerSearch label="Player 2" value={player2} onChange={setPlayer2} />
          <label>
            Format
            <select value={bestOf} onChange={e => setBestOf(parseInt(e.target.value))}>
              {[5, 7, 9, 11, 13, 17, 19, 25, 35].map(n => (
                <option key={n} value={n}>Best of {n}</option>
              ))}
            </select>
          </label>
          <button onClick={handlePredict} disabled={loading} style={{ alignSelf: 'flex-end' }}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </div>
      </div>

      {prediction && (
        <>
          <div className="card">
            <h2>ELO Prediction</h2>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <span><strong>{prediction.player1}</strong> ({prediction.elo.player1_rating})</span>
              <span><strong>{prediction.player2}</strong> ({prediction.elo.player2_rating})</span>
            </div>
            <div className="prob-bar">
              <div className="prob-bar-p1" style={{ width: `${prediction.elo.match_win_prob * 100}%` }}>
                {(prediction.elo.match_win_prob * 100).toFixed(1)}%
              </div>
              <div className="prob-bar-p2" style={{ width: `${(1 - prediction.elo.match_win_prob) * 100}%` }}>
                {((1 - prediction.elo.match_win_prob) * 100).toFixed(1)}%
              </div>
            </div>
            <p style={{ color: '#888', fontSize: '0.85rem' }}>
              Frame win probability: {(prediction.elo.frame_win_prob * 100).toFixed(1)}%
            </p>
          </div>

          <div className="card">
            <h2>Glicko-2 Prediction</h2>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <span>
                <strong>{prediction.player1}</strong> ({prediction.glicko2.player1_rating})
                <span className="badge badge-blue" style={{ marginLeft: 6 }}>RD {prediction.glicko2.player1_rd}</span>
              </span>
              <span>
                <span className="badge badge-blue" style={{ marginRight: 6 }}>RD {prediction.glicko2.player2_rd}</span>
                <strong>{prediction.player2}</strong> ({prediction.glicko2.player2_rating})
              </span>
            </div>
            <div className="prob-bar">
              <div className="prob-bar-p1" style={{ width: `${prediction.glicko2.match_win_prob * 100}%` }}>
                {(prediction.glicko2.match_win_prob * 100).toFixed(1)}%
              </div>
              <div className="prob-bar-p2" style={{ width: `${(1 - prediction.glicko2.match_win_prob) * 100}%` }}>
                {((1 - prediction.glicko2.match_win_prob) * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {prediction.score_probabilities && prediction.score_probabilities.length > 0 && (
            <div className="card">
              <h2>Score Probabilities</h2>
              <table>
                <thead>
                  <tr>
                    <th>Score</th>
                    <th>Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {prediction.score_probabilities
                    .filter(s => s.p2_prob !== undefined || s.p1_prob > 0.01)
                    .sort((a, b) => (b.p2_prob || (1 - b.p1_prob)) - (a.p2_prob || (1 - a.p1_prob)))
                    .slice(0, 10)
                    .map((s, i) => (
                      <tr key={i}>
                        <td style={{ fontWeight: 500 }}>{s.score}</td>
                        <td>{((s.p2_prob || s.p1_prob) * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </>
  )
}
