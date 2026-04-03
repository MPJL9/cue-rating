import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getRecentMatches } from '../api'

export default function RecentMatches() {
  const [tournaments, setTournaments] = useState([])
  const [loading, setLoading] = useState(true)
  const [expandedId, setExpandedId] = useState(null)

  useEffect(() => {
    getRecentMatches(30)
      .then(data => setTournaments(data.tournaments))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading tournaments...</div>

  const totalMatches = tournaments.reduce((s, t) => s + t.n_matches, 0)
  const totalUpsets = tournaments.reduce((s, t) => s + t.upsets, 0)
  const avgAccuracy = tournaments.length > 0
    ? (tournaments.reduce((s, t) => s + t.accuracy, 0) / tournaments.length * 100).toFixed(1)
    : 0

  return (
    <>
      <div className="page-header">
        <h1>Recent Tournaments</h1>
        <p>Click a tournament to see individual matches with ELO predictions</p>
      </div>

      <div className="grid-3" style={{ marginBottom: '1rem' }}>
        <div className="stat">
          <div className="value">{tournaments.length}</div>
          <div className="label">Tournaments</div>
        </div>
        <div className="stat">
          <div className="value">{totalMatches}</div>
          <div className="label">Total matches</div>
        </div>
        <div className="stat">
          <div className="value">{avgAccuracy}%</div>
          <div className="label">Avg ELO accuracy</div>
        </div>
      </div>

      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Tournament ID</th>
              <th>Year</th>
              <th>Matches</th>
              <th>ELO Accuracy</th>
              <th>Upsets</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {tournaments.map(t => (
              <React.Fragment key={t.tournament_id}>
                <tr
                  onClick={() => setExpandedId(expandedId === t.tournament_id ? null : t.tournament_id)}
                  style={{ cursor: 'pointer' }}
                >
                  <td style={{ fontWeight: 500 }}>#{t.tournament_id}</td>
                  <td>{t.year}</td>
                  <td>{t.n_matches}</td>
                  <td>
                    <span className={`badge ${t.accuracy > 0.7 ? 'badge-green' : t.accuracy > 0.6 ? 'badge-blue' : 'badge-gray'}`}>
                      {(t.accuracy * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td>
                    {t.upsets > 0 && (
                      <span className="badge badge-red">{t.upsets}</span>
                    )}
                  </td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                    {expandedId === t.tournament_id ? '▲ collapse' : '▼ expand'}
                  </td>
                </tr>
                {expandedId === t.tournament_id && (
                  <tr>
                    <td colSpan={6} style={{ padding: 0, background: 'var(--bg)' }}>
                      <MatchTable matches={t.matches} />
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function MatchTable({ matches }) {
  return (
    <table style={{ margin: '0.5rem 1rem 1rem', width: 'calc(100% - 2rem)' }}>
      <thead>
        <tr>
          <th>Winner</th>
          <th>Score</th>
          <th>Loser</th>
          <th>Format</th>
          <th>ELO Prediction</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {matches.map((m, i) => (
          <tr key={i}>
            <td>
              <Link to={`/player/${encodeURIComponent(m.player1)}`}>{m.player1}</Link>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginLeft: 6 }}>
                {m.p1_elo}
              </span>
            </td>
            <td style={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
              {m.score1}-{m.score2}
            </td>
            <td>
              <Link to={`/player/${encodeURIComponent(m.player2)}`}>{m.player2}</Link>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginLeft: 6 }}>
                {m.p2_elo}
              </span>
            </td>
            <td style={{ color: 'var(--text-muted)' }}>BO{m.best_of}</td>
            <td>
              <span style={{
                color: m.elo_win_prob > 0.5 ? 'var(--green)' : 'var(--text-muted)',
                fontWeight: 500,
              }}>
                {(m.elo_win_prob * 100).toFixed(0)}%
              </span>
            </td>
            <td>
              {m.upset && <span className="badge badge-red">UPSET</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
