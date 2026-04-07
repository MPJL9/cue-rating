import React, { useState, useEffect, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { searchPlayers, simulateTournament, simulateBracket, ServerLoadingError } from '../api'

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
  "Top 4": ["Judd Trump", "Mark Selby", "John Higgins", "Ronnie O'Sullivan"],
  "Top 8": ["Judd Trump", "Mark Selby", "John Higgins", "Ronnie O'Sullivan",
             "Kyren Wilson", "Mark Allen", "Neil Robertson", "Barry Hawkins"],
  "Top 16": ["Judd Trump", "Mark Selby", "John Higgins", "Ronnie O'Sullivan",
              "Kyren Wilson", "Mark Allen", "Neil Robertson", "Barry Hawkins",
              "Mark Williams", "Shaun Murphy", "Ding Junhui", "Yan Bingtao",
              "Zhao Xintong", "Luca Brecel", "Wu Yize", "Xiao Guodong"],
}

export default function Simulator() {
  const [players, setPlayers] = useState([])
  const [bestOf, setBestOf] = useState(9)
  const [mcResult, setMcResult] = useState(null)
  const [bracket, setBracket] = useState(null)
  const [formatType, setFormatType] = useState('ranking_event')
  const [loading, setLoading] = useState(false)
  const [serverLoading, setServerLoading] = useState(false)

  const addPlayer = useCallback((name) => {
    if (!players.includes(name)) setPlayers(prev => [...prev, name])
  }, [players])

  const removePlayer = useCallback((name) => {
    setPlayers(prev => prev.filter(p => p !== name))
  }, [])

  const handleSimulate = useCallback(() => {
    if (players.length < 2) return
    setLoading(true)
    setBracket(null)
    setMcResult(null)
    Promise.all([
      simulateBracket(players, bestOf, formatType),
      simulateTournament(players, bestOf, 10000, formatType),
    ])
      .then(([b, mc]) => { setBracket(b); setMcResult(mc) })
      .catch(e => {
        if (e instanceof ServerLoadingError) setServerLoading(true)
        else console.error(e)
      })
      .finally(() => setLoading(false))
  }, [players, bestOf])

  const handleResimulate = useCallback(() => {
    if (players.length < 2) return
    setLoading(true)
    simulateBracket(players, bestOf, formatType)
      .then(setBracket)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [players, bestOf])

  if (serverLoading) {
    return <div className="loading">Server is computing ratings... Please wait ~45s and refresh.</div>
  }

  const chartData = mcResult?.players?.map(p => ({
    name: p.name.split(' ').pop(),
    fullName: p.name,
    winProb: Math.round(p.win_prob * 1000) / 10,
  })) || []

  return (
    <>
      <div className="page-header">
        <h1>Tournament Simulator</h1>
        <p>Simulate a knockout bracket and estimate win probabilities via Monte Carlo</p>
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
            Tournament Format
            <select value={formatType} onChange={e => setFormatType(e.target.value)}
              style={{ width: 220 }}>
              <option value="world_championship">World Championship</option>
              <option value="ranking_event">Ranking Event</option>
              <option value="masters">Masters / Invitational</option>
              <option value="uniform">Custom (same every round)</option>
            </select>
          </label>
          {formatType === 'uniform' && (
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem',
              fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Best of
              <select value={bestOf} onChange={e => setBestOf(parseInt(e.target.value))}
                style={{ width: 100 }}>
                {[5, 7, 9, 11, 13, 17, 19, 25, 35].map(n => (
                  <option key={n} value={n}>BO{n}</option>
                ))}
              </select>
            </label>
          )}
          <button onClick={handleSimulate} disabled={loading || players.length < 2}>
            {loading ? 'Simulating...' : `Simulate (${players.length} players)`}
          </button>
        </div>

        {formatType !== 'uniform' && (
          <FormatPreview formatType={formatType} nPlayers={players.length} />
        )}

        <div className="player-chips">
          {players.map(name => (
            <div key={name} className="chip">
              {name}
              <button onClick={() => removePlayer(name)}>&times;</button>
            </div>
          ))}
        </div>
      </div>

      {bracket && (
        <div className="card">
          <div className="card-header">
            <h2>
              Simulated Bracket
              {bracket.champion && (
                <span style={{ color: 'var(--green)', marginLeft: '0.75rem', fontWeight: 400, fontSize: '0.9rem' }}>
                  Champion: {bracket.champion}
                </span>
              )}
            </h2>
            <button className="btn-outline" onClick={handleResimulate} disabled={loading}>
              Re-roll
            </button>
          </div>
          <BracketView bracket={bracket} />
        </div>
      )}

      {mcResult && (
        <>
          <div className="card">
            <div className="card-header">
              <h2>Win Probability (10,000 simulations)</h2>
            </div>
            <ResponsiveContainer width="100%" height={Math.max(200, mcResult.players.length * 36)}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 80 }}>
                <XAxis type="number" stroke="#71717a" fontSize={12}
                  tickFormatter={v => `${v}%`} domain={[0, 'auto']} />
                <YAxis type="category" dataKey="name" stroke="#71717a" fontSize={12}
                  width={80} interval={0} />
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
            <div className="card-header"><h2>Round-by-Round Advancement</h2></div>
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Rating</th>
                  {(mcResult.round_info || mcResult.round_names.map(r => ({name: r}))).map(r => (
                  <th key={r.name || r}>{r.name || r}{r.best_of ? ` (BO${r.best_of})` : ''}</th>
                ))}
                  <th>Win %</th>
                </tr>
              </thead>
              <tbody>
                {mcResult.players.map((p, i) => (
                  <tr key={p.name}>
                    <td style={{ fontWeight: 500 }}>{p.name}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{p.elo_rating}</td>
                    {mcResult.round_names.map(r => (
                      <td key={r} style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {(p.rounds[r] * 100).toFixed(1)}%
                      </td>
                    ))}
                    <td>
                      <span className={`badge ${i === 0 ? 'badge-green' : i < 3 ? 'badge-blue' : 'badge-gray'}`}>
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

// Backend format presets mirrored here (dist_from_final -> best_of)
const FORMAT_PRESETS = {
  world_championship: {
    label: 'World Championship',
    // 0=final, 1=semi, 2=QF, etc.
    bo: { 0: 35, 1: 33, 2: 25, 3: 25, 4: 19 },
    fallback: 19,
  },
  ranking_event: {
    label: 'Ranking Event (e.g. UK Championship)',
    bo: { 0: 19, 1: 11, 2: 11, 3: 11, 4: 7, 5: 7, 6: 7 },
    fallback: 7,
  },
  masters: {
    label: 'Masters / Invitational',
    bo: { 0: 19, 1: 11, 2: 11, 3: 11 },
    fallback: 11,
  },
}

function FormatPreview({ formatType, nPlayers }) {
  const fmt = FORMAT_PRESETS[formatType]
  if (!fmt) return null

  let bracketSize = 1
  const n = Math.max(nPlayers, 2)
  while (bracketSize < n) bracketSize *= 2
  let nRounds = 0
  let t = bracketSize
  while (t > 1) { nRounds++; t /= 2 }

  const roundNames = Array.from({ length: nRounds }, (_, i) => `Round ${i + 1}`)
  if (nRounds >= 1) roundNames[nRounds - 1] = 'Final'
  if (nRounds >= 2) roundNames[nRounds - 2] = 'Semi-Final'
  if (nRounds >= 3) roundNames[nRounds - 3] = 'Quarter-Final'

  return (
    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '0.5rem 0',
      display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
      {roundNames.map((name, idx) => {
        const distFromFinal = nRounds - 1 - idx
        const bo = fmt.bo[distFromFinal] ?? fmt.fallback
        return (
          <React.Fragment key={name}>
            {idx > 0 && <span style={{ color: '#444' }}>→</span>}
            <span style={{ background: 'var(--bg-hover)', padding: '0.15rem 0.5rem',
              borderRadius: 4 }}>
              {name} <span style={{ color: 'var(--green)', fontWeight: 600 }}>BO{bo}</span>
            </span>
          </React.Fragment>
        )
      })}
    </div>
  )
}

/* ── Bracket Visualization ── */

function BracketView({ bracket }) {
  const { rounds } = bracket
  if (!rounds || rounds.length === 0) return null

  return (
    <div className="bracket-scroll">
      <div className="bracket">
        {rounds.map((round, rIdx) => (
          <div className="bracket-round" key={rIdx}>
            <div className="bracket-round-name">
              {round.name}
              {round.best_of && <span style={{ fontWeight: 400 }}> (BO{round.best_of})</span>}
            </div>
            <div className="bracket-matches">
              {round.matches.filter(m => m.player1 !== 'BYE' && m.player2 !== 'BYE').map((m, mIdx) => (
                <BracketMatch key={mIdx} match={m} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function BracketMatch({ match }) {
  const { player1, player2, score1, score2, winner } = match

  return (
    <div className="bracket-match">
      <div className={`bracket-slot ${winner === player1 ? 'winner' : 'loser'}`}>
        <span className="bracket-name">{player1}</span>
        <span className="bracket-score">{score1}</span>
      </div>
      <div className={`bracket-slot ${winner === player2 ? 'winner' : 'loser'}`}>
        <span className="bracket-name">{player2}</span>
        <span className="bracket-score">{score2}</span>
      </div>
    </div>
  )
}
