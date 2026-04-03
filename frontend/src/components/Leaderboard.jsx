import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getRatings } from '../api'

export default function Leaderboard() {
  const [system, setSystem] = useState('elo')
  const [players, setPlayers] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    getRatings(system, 100)
      .then(data => setPlayers(data.players))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [system])

  return (
    <>
      <div className="card">
        <h2>Player Rankings</h2>
        <div className="tabs">
          <button
            className={`tab ${system === 'elo' ? 'active' : ''}`}
            onClick={() => setSystem('elo')}
          >
            ELO Rating
          </button>
          <button
            className={`tab ${system === 'glicko2' ? 'active' : ''}`}
            onClick={() => setSystem('glicko2')}
          >
            Glicko-2 Rating
          </button>
        </div>

        {loading ? (
          <div className="loading">Loading ratings...</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Player</th>
                <th>Rating</th>
                {system === 'glicko2' && <th>RD</th>}
                <th>Matches</th>
                <th>Win Rate</th>
              </tr>
            </thead>
            <tbody>
              {players.map(p => (
                <tr key={p.name}>
                  <td>{p.rank}</td>
                  <td>
                    <Link to={`/player/${encodeURIComponent(p.name)}`}>
                      {p.name}
                    </Link>
                  </td>
                  <td style={{ fontWeight: 600 }}>{p.rating}</td>
                  {system === 'glicko2' && (
                    <td>
                      <span className={`badge ${p.rd < 60 ? 'badge-green' : p.rd < 100 ? 'badge-blue' : 'badge-gray'}`}>
                        {p.rd}
                      </span>
                    </td>
                  )}
                  <td>{p.matches_played}</td>
                  <td>{(p.win_rate * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}
