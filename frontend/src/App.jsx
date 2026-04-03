import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import Leaderboard from './components/Leaderboard'
import PlayerProfile from './components/PlayerProfile'
import Predictor from './components/Predictor'
import Comparison from './components/Comparison'

export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <span className="nav-brand">Snooker Ratings</span>
        <NavLink to="/" end>Leaderboard</NavLink>
        <NavLink to="/predict">Predict</NavLink>
        <NavLink to="/comparison">ELO vs Glicko-2</NavLink>
      </nav>
      <main className="main">
        <Routes>
          <Route path="/" element={<Leaderboard />} />
          <Route path="/player/:name" element={<PlayerProfile />} />
          <Route path="/predict" element={<Predictor />} />
          <Route path="/comparison" element={<Comparison />} />
        </Routes>
      </main>
    </div>
  )
}
