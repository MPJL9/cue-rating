import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import Leaderboard from './components/Leaderboard'
import PlayerProfile from './components/PlayerProfile'
import Predictor from './components/Predictor'
import Simulator from './components/Simulator'
import RecentMatches from './components/RecentMatches'
import PrimeTimes from './components/PrimeTimes'
import Methodology from './components/Methodology'

export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <NavLink to="/" className="nav-brand">Snooker Ratings</NavLink>
        <div className="nav-links">
          <NavLink to="/" end>Rankings</NavLink>
          <NavLink to="/matches">Matches</NavLink>
          <NavLink to="/predict">Predict</NavLink>
          <NavLink to="/simulate">Simulate</NavLink>
          <NavLink to="/primes">Prime Times</NavLink>
          <NavLink to="/methodology">Methodology</NavLink>
        </div>
      </nav>
      <main className="main">
        <Routes>
          <Route path="/" element={<Leaderboard />} />
          <Route path="/player/:name" element={<PlayerProfile />} />
          <Route path="/matches" element={<RecentMatches />} />
          <Route path="/predict" element={<Predictor />} />
          <Route path="/simulate" element={<Simulator />} />
          <Route path="/primes" element={<PrimeTimes />} />
          <Route path="/methodology" element={<Methodology />} />
        </Routes>
      </main>
      <footer className="footer">
        <span>117,530 matches &middot; 1,108 tournaments &middot; 1982-2026</span>
        <span>Data last updated: April 7, 2026 &middot; ELO + Glicko-2</span>
      </footer>
    </div>
  )
}
