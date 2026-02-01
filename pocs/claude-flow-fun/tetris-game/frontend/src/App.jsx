import { Routes, Route, Link } from 'react-router-dom'
import TetrisBoard from './components/TetrisBoard'
import AdminPanel from './components/AdminPanel'

function App() {
  return (
    <div className="app">
      <nav className="main-nav">
        <Link to="/" className="nav-link">Game</Link>
        <Link to="/admin" className="nav-link">Admin</Link>
      </nav>
      <Routes>
        <Route path="/" element={<TetrisBoard />} />
        <Route path="/admin" element={<AdminPanel />} />
      </Routes>
    </div>
  )
}

export default App
