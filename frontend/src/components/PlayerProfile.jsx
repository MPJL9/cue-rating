import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { getPlayer, getPlayerHistory } from '../api'

export default function PlayerProfile() {
  const { name } = useParams()
  const [player, setPlayer] = useState(null)
  const [history, setHistory] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const decoded = decodeURIComponent(name)
    setLoading(true)
    Promise.all([getPlayer(decoded), getPlayerHistory(decoded)])
      .then(([p, h]) => { setPlayer(p); setHistory(h) })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [name])

  if (loading) return <div className="loading">Loading player...</div>
  if (!player) return <div className="card"><p>Player not found.</p></div>

  const elo = player.elo || {}
  const g2 = player.glicko2 || {}

  return (
    <>
      <div className="page-header">
        <h1>{player.name}</h1>
      </div>

      <div className="grid-4">
        <div className="stat">
          <div className="value">{elo.rating || '—'}</div>
          <div className="label">ELO</div>
        </div>
        <div className="stat">
          <div className="value">{g2.rating || '—'}</div>
          <div className="label">Glicko-2</div>
        </div>
        {g2.rd && (
          <div className="stat">
            <div className="value">{g2.rd}</div>
            <div className="label">RD (Uncertainty)</div>
          </div>
        )}
        <div className="stat">
          <div className="value">{elo.matches_played || 0}</div>
          <div className="label">Matches</div>
        </div>
        <div className="stat">
          <div className="value">{((elo.win_rate || 0) * 100).toFixed(1)}%</div>
          <div className="label">Win Rate</div>
        </div>
        <div className="stat">
          <div className="value">{((elo.frame_win_rate || 0) * 100).toFixed(1)}%</div>
          <div className="label">Frame Win Rate</div>
        </div>
      </div>

      {history && history.elo_history && history.elo_history.length > 2 && (
        <div className="card" style={{ marginTop: '1rem' }}>
          <div className="card-header">
            <h2>ELO Rating History</h2>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={history.elo_history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="year" stroke="#71717a" fontSize={12} />
              <YAxis stroke="#71717a" fontSize={12} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 6 }}
                labelStyle={{ color: '#71717a' }}
                itemStyle={{ color: '#22c55e' }}
              />
              <Line type="monotone" dataKey="rating" stroke="#22c55e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="card" style={{ marginTop: '1rem' }}>
        <div className="card-header">
          <h2>Recent Matches</h2>
        </div>
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
                  <Link to={`/player/${encodeURIComponent(m.opponent)}`}>{m.opponent}</Link>
                </td>
                <td style={{ fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{m.score}</td>
                <td>
                  <span className={`badge ${m.won ? 'badge-green' : 'badge-red'}`}>
                    {m.won ? 'W' : 'L'}
                  </span>
                </td>
                <td style={{ color: 'var(--text-muted)' }}>BO{m.best_of}</td>
                <td style={{ color: 'var(--text-muted)' }}>{m.year}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
