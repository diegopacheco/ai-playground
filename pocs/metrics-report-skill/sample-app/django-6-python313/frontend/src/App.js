import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import HomePage from './pages/HomePage';
import SimulationPage from './pages/SimulationPage';
import ResultsPage from './pages/ResultsPage';
import AboutPage from './pages/AboutPage';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [inputData, setInputData] = useState(null);

  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-brand">RetireSmart</div>
          <div className="nav-links">
            <NavLink to="/" end className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>Home</NavLink>
            <NavLink to="/simulate" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>Simulate</NavLink>
            <NavLink to="/results" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>Results</NavLink>
            <NavLink to="/about" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>About</NavLink>
          </div>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/simulate" element={<SimulationPage setResults={setResults} setInputData={setInputData} />} />
            <Route path="/results" element={<ResultsPage results={results} inputData={inputData} />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
