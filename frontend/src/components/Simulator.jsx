import React, { useState, useEffect, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'
import { searchPlayers, simulateTournament } from '../api'

function PlayerSearch({ onAdd }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (query.length < 2) { setResults([]); return }
    const timer = setTimeout(() => {
      searchPlayers(query).then(d => setResults(d.results.slice(0, 8))).catch(() => {})
    }, 300)
    return () => clearTimeout(timer)
  }, [query])

  return (
    <div style={{ position: 'relative', maxWidth: 300 }}>
      <input
        value={query}
        onChange={e => { setQuery(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        placeholder="Search and add players..."
      />
      {open && results.length > 0 && (
        <div className="autocomplete">
          {results.map(r => (
            <div key={r.name} className="autocomplete-item"
              onClick={() => { onAdd(r.name); setQuery(''); setOpen(false) }}>
              {r.name} <span style={{ color: 'var(--text-muted)' }}>({r.elo_rating})</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const PRESETS = {
  "Top 8": ["Judd Trump", "Mark Selby", "John Higgins", "Ronnie O'Sullivan",
             "Kyren Wilson", "Mark Allen", "Neil Robertson", "Barry Hawkins"],
  "Top 4": ["Judd Trump", "Mark Selby", "John Higgins", "Ronnie O'Sullivan"],
}

export default function Simulator() {
  const [players, setPlayers] = useState([])
  const [bestOf, setBestOf] = useState(17)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const addPlayer = useCallback((name) => {
    if (!players.includes(name)) setPlayers(prev => [...prev, name])
  }, [players])

  const removePlayer = useCallback((name) => {
    setPlayers(prev => prev.filter(p => p !== name))
  }, [])

  const handleSimulate = useCallback(() => {
    if (players.length < 2) return
    setLoading(true)
    simulateTournament(players, bestOf, 10000)
      .then(setResult)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [players, bestOf])

  const chartData = result?.players?.map(p => ({
    name: p.name.split(' ').pop(),  // Last name for chart
    fullName: p.name,
    winProb: Math.round(p.win_prob * 1000) / 10,
    rating: p.elo_rating,
  })) || []

  return (
    <>
      <div className="page-header">
        <h1>Tournament Simulator</h1>
        <p>Monte Carlo simulation of a single-elimination bracket (10,000 runs)</p>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Build Your Bracket</h2>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {Object.entries(PRESETS).map(([name, list]) => (
              <button key={name} className="btn-outline" onClick={() => setPlayers(list)}>
                {name}
              </button>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <PlayerSearch onAdd={addPlayer} />
          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem',
            fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Format
            <select value={bestOf} onChange={e => setBestOf(parseInt(e.target.value))}
              style={{ width: 120 }}>
              {[5, 7, 9, 11, 13, 17, 19, 25, 35].map(n => (
                <option key={n} value={n}>Best of {n}</option>
              ))}
            </select>
          </label>
          <button onClick={handleSimulate} disabled={loading || players.length < 2}>
            {loading ? 'Simulating...' : `Simulate (${players.length} players)`}
          </button>
        </div>

        <div className="player-chips">
          {players.map(name => (
            <div key={name} className="chip">
              {name}
              <button onClick={() => removePlayer(name)}>&times;</button>
            </div>
          ))}
        </div>
      </div>

      {result && (
        <>
          <div className="card">
            <div className="card-header">
              <h2>Win Probability</h2>
              <span className="badge badge-gray">{result.n_simulations.toLocaleString()} simulations</span>
            </div>
            <ResponsiveContainer width="100%" height={Math.max(250, result.players.length * 36)}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
                <XAxis type="number" stroke="#71717a" fontSize={12}
                  tickFormatter={v => `${v}%`} domain={[0, 'auto']} />
                <YAxis type="category" dataKey="name" stroke="#71717a" fontSize={12} width={80} />
                <Tooltip
                  contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 6 }}
                  formatter={(v, _, props) => [`${v}%`, props.payload.fullName]}
                />
                <Bar dataKey="winProb" radius={[0, 4, 4, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill={i === 0 ? '#22c55e' : i < 3 ? '#3b82f6' : '#52525b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <div className="card-header">
              <h2>Detailed Results</h2>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Rating</th>
                  {result.round_names.map(r => <th key={r}>{r}</th>)}
                  <th>Win %</th>
                </tr>
              </thead>
              <tbody>
                {result.players.map((p, i) => (
                  <tr key={p.name}>
                    <td style={{ fontWeight: 500 }}>{p.name}</td>
                    <td>{p.elo_rating}</td>
                    {result.round_names.map(r => (
                      <td key={r} style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {(p.rounds[r] * 100).toFixed(1)}%
                      </td>
                    ))}
                    <td>
                      <span className={`badge ${i === 0 ? 'badge-green' : i < 3 ? 'badge-blue' : 'badge-gray'}`}
                        style={{ fontWeight: 700 }}>
                        {(p.win_prob * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  )
}
