import { useState } from 'react'
import CoinFlipTab from './tabs/CoinFlipTab'
import DiceRollTab from './tabs/DiceRollTab'
import CLTTab from './tabs/CLTTab'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState(0)

  const tabs = [
    { label: 'Coin Flip Distribution', component: <CoinFlipTab /> },
    { label: 'Dice Roll Averages', component: <DiceRollTab /> },
    { label: 'Central Limit Theorem', component: <CLTTab /> },
  ]

  return (
    <div className="app">
      <header>
        <h1>Bell Curves</h1>
        <p className="subtitle">
          The Math That Explains Why Bell Curves Are Everywhere
        </p>
      </header>
      <nav className="tabs">
        {tabs.map((tab, i) => (
          <button
            key={i}
            className={activeTab === i ? 'active' : ''}
            onClick={() => setActiveTab(i)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <main>{tabs[activeTab].component}</main>
    </div>
  )
}

export default App
