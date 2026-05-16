import React, { useEffect, useState } from 'react';
import { useCopilotReadable, useCopilotAction } from '@copilotkit/react-core';
import { CopilotChat } from '@copilotkit/react-ui';
import StockWidget from './StockWidget.jsx';
import BranchMap from './BranchMap.jsx';

export default function App() {
  const [stocks, setStocks] = useState([]);
  const [selected, setSelected] = useState(null);
  const [branches, setBranches] = useState([]);
  const [userLocation, setUserLocation] = useState(null);

  useEffect(() => {
    fetch('/api/stocks').then(r => r.json()).then(setStocks).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selected) {
      setBranches([]);
      return;
    }
    fetch(`/api/branches/${selected}`).then(r => r.json()).then(setBranches).catch(() => {});
  }, [selected]);

  useEffect(() => {
    if (!navigator.geolocation) {
      setUserLocation({ lat: 37.7749, lng: -122.4194 });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      p => setUserLocation({ lat: p.coords.latitude, lng: p.coords.longitude }),
      () => setUserLocation({ lat: 37.7749, lng: -122.4194 })
    );
  }, []);

  useCopilotReadable({ description: 'List of all available stocks with prices', value: stocks });
  useCopilotReadable({ description: 'Currently selected stock symbol', value: selected });
  useCopilotReadable({ description: 'Branch locations for the currently selected company', value: branches });
  useCopilotReadable({ description: 'User current geolocation coordinates', value: userLocation });

  useCopilotAction({
    name: 'showStock',
    description: 'Show the stock price widget and branch map for a given company. Call this whenever the user asks about a stock or company.',
    parameters: [
      { name: 'symbol', type: 'string', description: 'Stock ticker symbol such as AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA', required: true }
    ],
    handler: async ({ symbol }) => {
      const sym = symbol.toUpperCase();
      setSelected(sym);
      return `Showing ${sym} widget and branches on the map.`;
    }
  });

  useCopilotAction({
    name: 'hideStock',
    description: 'Hide the stock widget and map, returning to the chat-only view.',
    parameters: [],
    handler: async () => {
      setSelected(null);
      return 'Closed the stock widget.';
    }
  });

  const currentStock = stocks.find(s => s.symbol === selected);
  const showWidgets = Boolean(selected);

  return (
    <div className="app">
      <header>
        <h1>Stock Insight Copilot</h1>
        <p className="subtitle">Ask the assistant about a stock to see its price and nearest branch.</p>
      </header>
      <div className={`layout ${showWidgets ? 'with-widgets' : 'chat-only'}`}>
        <div className="chat-panel">
          <CopilotChat
            instructions="You help the user explore stock prices and find the nearest company branches. Whenever the user asks about a stock or company (Apple, Google, Microsoft, Amazon, Tesla, NVIDIA, or their tickers AAPL/GOOGL/MSFT/AMZN/TSLA/NVDA), call the showStock action with the ticker. If they ask to close or hide it, call hideStock."
            labels={{
              title: 'Stock Assistant',
              initial: "Hi! Ask me about a stock like \"show me Apple\" or \"what's Tesla's price?\" and I'll bring up the widget."
            }}
          />
        </div>
        {showWidgets && (
          <div className="widgets-panel">
            <div className="widget-header">
              <span>Showing <strong>{selected}</strong></span>
              <button className="close-btn" onClick={() => setSelected(null)}>Close</button>
            </div>
            <StockWidget stock={currentStock} />
            <BranchMap branches={branches} userLocation={userLocation} symbol={selected} />
          </div>
        )}
      </div>
    </div>
  );
}
