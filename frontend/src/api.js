const API_BASE = '/api'

class ServerLoadingError extends Error {
  constructor() {
    super('Server is computing ratings. Please wait...')
    this.name = 'ServerLoadingError'
  }
}

async function fetchJSON(path, retries = 3) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    const res = await fetch(`${API_BASE}${path}`)
    if (res.status === 503) {
      if (attempt < retries) {
        // Wait and retry — server is still computing ratings
        await new Promise(r => setTimeout(r, 5000))
        continue
      }
      throw new ServerLoadingError()
    }
    if (!res.ok) throw new Error(`API error: ${res.status}`)
    return res.json()
  }
}

async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (res.status === 503) throw new ServerLoadingError()
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export { ServerLoadingError }

export const getRatings = (system = 'elo', top = 100) =>
  fetchJSON(`/ratings?system=${system}&top=${top}`)

export const getPlayer = (name) =>
  fetchJSON(`/player/${encodeURIComponent(name)}`)

export const getPlayerHistory = (name) =>
  fetchJSON(`/player/${encodeURIComponent(name)}/history`)

export const predictMatch = (player1, player2, bestOf) =>
  postJSON('/predict', { player1, player2, best_of: bestOf })

export const getComparison = () => fetchJSON('/comparison')

export const searchPlayers = (query) =>
  fetchJSON(`/search?q=${encodeURIComponent(query)}`)

export const getStats = () => fetchJSON('/stats')

export const simulateTournament = (players, bestOf, simulations = 10000) =>
  postJSON('/simulate', { players, best_of: bestOf, simulations })

export const getRecentMatches = (limit = 50) =>
  fetchJSON(`/matches/recent?limit=${limit}`)

export const getPrimeTimes = (minMatches = 200) =>
  fetchJSON(`/prime-times?min_matches=${minMatches}`)
