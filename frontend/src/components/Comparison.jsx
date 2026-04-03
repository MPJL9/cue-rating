import React, { useState, useEffect } from 'react'
import { getComparison } from '../api'

export default function Comparison() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getComparison()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading comparison...</div>
  if (!data) return null

  return (
    <>
      <div className="card">
        <h2>ELO vs Glicko-2 Comparison</h2>

        <div className="metric-grid">
          <div className="metric-card">
            <div className="value">{data.total_players}</div>
            <div className="label">Total Players</div>
          </div>
          <div className="metric-card">
            <div className="value">{data.top20_overlap}/20</div>
            <div className="label">Top-20 Overlap</div>
          </div>
        </div>

        <div style={{ marginTop: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>System Strengths</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div style={{ padding: '1rem', background: '#0a1a0a', borderRadius: 6, border: '1px solid #1a3a1a' }}>
              <strong style={{ color: '#22c55e' }}>ELO</strong>
              <p style={{ color: '#888', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                {data.summary?.elo || 'Better accuracy and log-loss'}
              </p>
            </div>
            <div style={{ padding: '1rem', background: '#0a0a1a', borderRadius: 6, border: '1px solid #1a1a3a' }}>
              <strong style={{ color: '#60a5fa' }}>Glicko-2</strong>
              <p style={{ color: '#888', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                {data.summary?.glicko2 || 'Better calibration and uncertainty estimates'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {data.biggest_disagreements && data.biggest_disagreements.length > 0 && (
        <div className="card">
          <h2>Biggest Rating Disagreements</h2>
          <p style={{ color: '#888', fontSize: '0.85rem', marginBottom: '1rem' }}>
            Players where ELO and Glicko-2 rankings diverge the most
          </p>
          <table>
            <thead>
              <tr>
                <th>Player</th>
                <th>ELO Rank</th>
                <th>Glicko-2 Rank</th>
                <th>Difference</th>
              </tr>
            </thead>
            <tbody>
              {data.biggest_disagreements.map((d, i) => (
                <tr key={i}>
                  <td>{d.name}</td>
                  <td>#{d.elo_rank}</td>
                  <td>#{d.glicko2_rank}</td>
                  <td>
                    <span className={`badge ${d.rank_diff > 50 ? 'badge-gray' : 'badge-blue'}`}>
                      {d.rank_diff} places
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <h2>Methodology</h2>
        <div style={{ color: '#aaa', fontSize: '0.9rem', lineHeight: 1.8 }}>
          <p><strong>ELO Rating</strong> (K=9.77, divisor=327.15 — MLE-optimized)</p>
          <p>Frame-weighted update: R<sub>new</sub> = R + K &times; total_frames &times; (S - E)</p>
          <p>Expected frame win rate: E = 1 / (1 + e<sup>(R2-R1)/327</sup>)</p>
          <p>Match win probability via binomial formula from frame probability.</p>
          <br />
          <p><strong>Glicko-2</strong> (tau=1.488 — MLE-optimized)</p>
          <p>Extends ELO with Rating Deviation (uncertainty) and Volatility (consistency).</p>
          <p>RD decreases with more games, increases with inactivity.</p>
          <p>Players returning from long breaks have higher RD, making predictions less confident.</p>
          <br />
          <p>Parameters optimized via Maximum Likelihood Estimation on 34,521 matches across 300 tournaments.</p>
        </div>
      </div>
    </>
  )
}
