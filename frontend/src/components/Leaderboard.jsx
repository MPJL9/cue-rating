import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getRatings, ServerLoadingError } from '../api'

export default function Leaderboard() {
  const [system, setSystem] = useState('elo')
  const [players, setPlayers] = useState([])
  const [loading, setLoading] = useState(true)
  const [serverLoading, setServerLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    setServerLoading(false)
    getRatings(system, 100)
      .then(data => setPlayers(data.players))
      .catch(e => {
        if (e instanceof ServerLoadingError) setServerLoading(true)
        else console.error(e)
      })
      .finally(() => setLoading(false))
  }, [system])

  if (serverLoading) return <ServerStarting />

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

function ServerStarting() {
  return (
    <div style={{ textAlign: 'center', padding: '4rem 2rem' }}>
      <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>Waking up the server...</div>
      <p style={{ color: 'var(--text-muted)', maxWidth: 500, margin: '0 auto', lineHeight: 1.8 }}>
        Computing ELO and Glicko-2 ratings for 117,530 professional snooker matches.
        This takes about 45 seconds on first visit. The page will refresh automatically.
      </p>
      <div style={{ marginTop: '2rem', color: 'var(--green)' }}>
        <div className="spinner" />
      </div>
    </div>
  )
}
