const API_BASE = '/api'

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function getRatings(system = 'elo', top = 50) {
  return fetchJSON(`/ratings?system=${system}&top=${top}`)
}

export async function getPlayer(name) {
  return fetchJSON(`/player/${encodeURIComponent(name)}`)
}

export async function predictMatch(player1, player2, bestOf) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player1, player2, best_of: bestOf }),
  })
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function getComparison() {
  return fetchJSON('/comparison')
}

export async function searchPlayers(query) {
  return fetchJSON(`/search?q=${encodeURIComponent(query)}`)
}

export async function getStats() {
  return fetchJSON('/stats')
}
