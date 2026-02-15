import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [status, setStatus] = useState<string>('loading...');

  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setStatus(data.status))
      .catch(() => setStatus('error'));
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>SnarkTank</h1>
        <p>Backend status: {status}</p>
      </header>
    </div>
  );
}

export default App;
