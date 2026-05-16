import React from 'react';

export default function StockWidget({ stock }) {
  if (!stock) return <div className="widget stock-widget"><div className="loading">Loading stock data...</div></div>;
  const positive = stock.change >= 0;
  const pct = (stock.change / stock.price * 100).toFixed(2);
  return (
    <div className="widget stock-widget">
      <div className="stock-header">
        <div>
          <div className="symbol">{stock.symbol}</div>
          <div className="name">{stock.name}</div>
        </div>
        <div className={`badge ${positive ? 'up' : 'down'}`}>{positive ? 'BULL' : 'BEAR'}</div>
      </div>
      <div className="price">${stock.price.toFixed(2)}</div>
      <div className={`change ${positive ? 'up' : 'down'}`}>
        <span className="arrow">{positive ? '▲' : '▼'}</span>
        <span>{Math.abs(stock.change).toFixed(2)} ({pct}%)</span>
      </div>
      <div className="sparkline">
        {Array.from({ length: 20 }).map((_, i) => {
          const seed = (stock.symbol.charCodeAt(0) + i) % 7;
          const h = 20 + (seed * 5 + i * 3) % 60;
          return <span key={i} className="bar" style={{ height: `${h}px`, background: positive ? '#22c55e' : '#ef4444' }} />;
        })}
      </div>
    </div>
  );
}
