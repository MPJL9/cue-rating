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
      <div className="page-header">
        <h1>Player Rankings</h1>
        <p>Professional snooker players rated by {system === 'elo' ? 'ELO' : 'Glicko-2'} system</p>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="tabs" style={{ marginBottom: 0, borderBottom: 'none' }}>
            <button className={`tab ${system === 'elo' ? 'active' : ''}`}
              onClick={() => setSystem('elo')}>ELO</button>
            <button className={`tab ${system === 'glicko2' ? 'active' : ''}`}
              onClick={() => setSystem('glicko2')}>Glicko-2</button>
          </div>
        </div>

        {loading ? (
          <div className="loading">Loading ratings...</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th style={{ width: 50 }}>#</th>
                <th>Player</th>
                <th>Rating</th>
                {system === 'glicko2' && <th>RD</th>}
                <th>Matches</th>
                <th>Won</th>
                <th>Win Rate</th>
              </tr>
            </thead>
            <tbody>
              {players.map(p => (
                <tr key={p.name}>
                  <td style={{ color: 'var(--text-muted)' }}>{p.rank}</td>
                  <td>
                    <Link to={`/player/${encodeURIComponent(p.name)}`}>{p.name}</Link>
                  </td>
                  <td style={{ fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>{p.rating}</td>
                  {system === 'glicko2' && (
                    <td>
                      <span className={`badge ${p.rd < 60 ? 'badge-green' : p.rd < 100 ? 'badge-blue' : 'badge-gray'}`}>
                        {Math.round(p.rd)}
                      </span>
                    </td>
                  )}
                  <td style={{ fontVariantNumeric: 'tabular-nums' }}>{p.matches_played}</td>
                  <td style={{ fontVariantNumeric: 'tabular-nums' }}>{p.matches_won}</td>
                  <td>
                    <span className={`badge ${p.win_rate > 0.6 ? 'badge-green' : p.win_rate > 0.5 ? 'badge-blue' : 'badge-gray'}`}>
                      {(p.win_rate * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}
