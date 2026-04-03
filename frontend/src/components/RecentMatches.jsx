import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getRecentMatches } from '../api'

export default function RecentMatches() {
  const [matches, setMatches] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getRecentMatches(100)
      .then(data => setMatches(data.matches))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading matches...</div>

  const upsets = matches.filter(m => m.upset).length
  const upsetRate = matches.length > 0 ? (upsets / matches.length * 100).toFixed(1) : 0

  return (
    <>
      <div className="page-header">
        <h1>Recent Matches</h1>
        <p>Latest professional snooker results with ELO predictions</p>
      </div>

      <div className="grid-3" style={{ marginBottom: '1rem' }}>
        <div className="stat">
          <div className="value">{matches.length}</div>
          <div className="label">Matches shown</div>
        </div>
        <div className="stat">
          <div className="value" style={{ color: 'var(--red)' }}>{upsets}</div>
          <div className="label">Upsets</div>
        </div>
        <div className="stat">
          <div className="value">{upsetRate}%</div>
          <div className="label">Upset rate</div>
        </div>
      </div>

      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Winner</th>
              <th>Score</th>
              <th>Loser</th>
              <th>Format</th>
              <th>ELO Prediction</th>
              <th>Year</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {matches.map((m, i) => (
              <tr key={i}>
                <td>
                  <Link to={`/player/${encodeURIComponent(m.player1)}`}>
                    {m.player1}
                  </Link>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginLeft: 6 }}>
                    {m.p1_elo}
                  </span>
                </td>
                <td style={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                  {m.score1}-{m.score2}
                </td>
                <td>
                  <Link to={`/player/${encodeURIComponent(m.player2)}`}>
                    {m.player2}
                  </Link>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginLeft: 6 }}>
                    {m.p2_elo}
                  </span>
                </td>
                <td style={{ color: 'var(--text-muted)' }}>BO{m.best_of}</td>
                <td>
                  <span style={{
                    color: m.elo_win_prob > 0.5 ? 'var(--green)' : 'var(--text-muted)',
                    fontWeight: 500, fontVariantNumeric: 'tabular-nums',
                  }}>
                    {(m.elo_win_prob * 100).toFixed(0)}%
                  </span>
                </td>
                <td style={{ color: 'var(--text-muted)' }}>{m.year}</td>
                <td>
                  {m.upset && (
                    <span className="badge badge-red">UPSET</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
