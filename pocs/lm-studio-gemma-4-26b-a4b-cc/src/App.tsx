import React from 'react';
import { createBrowserRouter, RouterProvider, Link, Outlet } from 'react-router-dom';
import { GamePage } from './pages/GamePage';
import { HistoryPage } from './pages/HistoryPage';
import { createRoot } from 'react-dom/client';

const router = createBrowserRouter([
  {
    path: '/',
    element: (
      <div>
        <nav style={{ padding: '10px', background: '#eee' }}>
          <Link to="/" style={{ marginRight: '10px' }}>Play</Link>
          <Link to="/history">History</Link>
        </nav>
        <hr />
        <Outlet />
      </div>
    ),
    children: [
      { index: true, element: <GamePage /> },
      { path: 'history', element: <HistoryPage /> },
    ],
  },
]);

export const App = () => {
  return <RouterProvider router={router} />;
};

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
