import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getPlayer } from '../api'

export default function PlayerProfile() {
  const { name } = useParams()
  const [player, setPlayer] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    getPlayer(decodeURIComponent(name))
      .then(setPlayer)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [name])

  if (loading) return <div className="loading">Loading player...</div>
  if (error) return <div className="card"><p>Player not found.</p></div>
  if (!player) return null

  const elo = player.elo || {}
  const g2 = player.glicko2 || {}

  return (
    <>
      <div className="card">
        <h2>{player.name}</h2>

        <div className="metric-grid">
          <div className="metric-card">
            <div className="value">{elo.rating || '—'}</div>
            <div className="label">ELO Rating</div>
          </div>
          <div className="metric-card">
            <div className="value">{g2.rating || '—'}</div>
            <div className="label">Glicko-2 Rating</div>
          </div>
          {g2.rd && (
            <div className="metric-card">
              <div className="value">{g2.rd}</div>
              <div className="label">Rating Deviation</div>
            </div>
          )}
          <div className="metric-card">
            <div className="value">{elo.matches_played || 0}</div>
            <div className="label">Matches Played</div>
          </div>
          <div className="metric-card">
            <div className="value">{((elo.win_rate || 0) * 100).toFixed(1)}%</div>
            <div className="label">Match Win Rate</div>
          </div>
          <div className="metric-card">
            <div className="value">{((elo.frame_win_rate || 0) * 100).toFixed(1)}%</div>
            <div className="label">Frame Win Rate</div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Recent Matches</h2>
        <table>
          <thead>
            <tr>
              <th>Opponent</th>
              <th>Score</th>
              <th>Result</th>
              <th>Format</th>
              <th>Year</th>
            </tr>
          </thead>
          <tbody>
            {(player.recent_matches || []).reverse().map((m, i) => (
              <tr key={i}>
                <td>
                  <Link to={`/player/${encodeURIComponent(m.opponent)}`}>
                    {m.opponent}
                  </Link>
                </td>
                <td style={{ fontWeight: 500 }}>{m.score}</td>
                <td>
                  <span className={`badge ${m.won ? 'badge-green' : 'badge-gray'}`}>
                    {m.won ? 'W' : 'L'}
                  </span>
                </td>
                <td>BO{m.best_of}</td>
                <td>{m.year}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
