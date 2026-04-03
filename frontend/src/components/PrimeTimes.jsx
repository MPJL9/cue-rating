import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { getPrimeTimes } from '../api'

export default function PrimeTimes() {
  const [players, setPlayers] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getPrimeTimes(200)
      .then(data => setPlayers(data.players))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading prime time data...</div>

  const top30 = players.slice(0, 30)
  const chartData = top30.map(p => ({
    name: p.name.split(' ').pop(),
    fullName: p.name,
    peak: Math.round(p.peak_rating),
    current: Math.round(p.current_rating),
    decline: Math.round(p.decline),
  }))

  return (
    <>
      <div className="page-header">
        <h1>Prime Times</h1>
        <p>Peak ELO rating and career trajectory for players with 200+ matches</p>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Peak Rating (Top 30)</h2>
        </div>
        <ResponsiveContainer width="100%" height={top30.length * 40 + 40}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 100, top: 10, bottom: 10 }}>
            <XAxis type="number" stroke="#71717a" fontSize={12} domain={[800, 'auto']} />
            <YAxis type="category" dataKey="name" stroke="#71717a" fontSize={11} width={100}
              interval={0} tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 6 }}
              formatter={(v, name, props) => {
                if (name === 'peak') return [`${v}`, `${props.payload.fullName} (Peak)`]
                return [`${v}`, `${props.payload.fullName} (Current)`]
              }}
            />
            <Bar dataKey="peak" fill="#22c55e" radius={[0, 4, 4, 0]} barSize={12} name="peak" />
            <Bar dataKey="current" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={12} name="current" />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: '1.5rem', justifyContent: 'center', marginTop: '0.5rem', fontSize: '0.8rem' }}>
          <span><span style={{ color: '#22c55e' }}>&#9632;</span> Peak rating</span>
          <span><span style={{ color: '#3b82f6' }}>&#9632;</span> Current rating</span>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Career Details</h2>
        </div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Player</th>
              <th>Peak Rating</th>
              <th>Peak Year</th>
              <th>Current</th>
              <th>Decline</th>
              <th>Career</th>
              <th>Matches</th>
              <th>Win Rate</th>
            </tr>
          </thead>
          <tbody>
            {top30.map((p, i) => (
              <tr key={p.name}>
                <td style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                <td>
                  <Link to={`/player/${encodeURIComponent(p.name)}`}>{p.name}</Link>
                </td>
                <td style={{ fontWeight: 700, color: 'var(--green)' }}>{Math.round(p.peak_rating)}</td>
                <td>{p.peak_year}</td>
                <td style={{ fontVariantNumeric: 'tabular-nums' }}>{Math.round(p.current_rating)}</td>
                <td>
                  {p.decline > 50 ? (
                    <span className="badge badge-red">-{Math.round(p.decline)}</span>
                  ) : p.decline > 0 ? (
                    <span className="badge badge-gray">-{Math.round(p.decline)}</span>
                  ) : (
                    <span className="badge badge-green">+{Math.round(-p.decline)}</span>
                  )}
                </td>
                <td style={{ color: 'var(--text-muted)' }}>
                  {p.career_start}–{p.career_end}
                </td>
                <td style={{ fontVariantNumeric: 'tabular-nums' }}>{p.matches_played}</td>
                <td>
                  <span className={`badge ${p.win_rate > 0.6 ? 'badge-green' : p.win_rate > 0.5 ? 'badge-blue' : 'badge-gray'}`}>
                    {(p.win_rate * 100).toFixed(1)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
