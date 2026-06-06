import React from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createRootRoute, createRoute, createRouter, RouterProvider, Outlet, Link } from '@tanstack/react-router';
import { Catalog } from './pages/Catalog';
import { Confirmation } from './pages/Confirmation';
import './styles.css';

const rootRoute = createRootRoute({
  component: () => (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">TAP&nbsp;Market</Link>
        <span className="tag">an agent shops here over Visa Trusted Agent Protocol</span>
      </header>
      <main className="content"><Outlet /></main>
    </div>
  )
});

const indexRoute = createRoute({ getParentRoute: () => rootRoute, path: '/', component: Catalog });
const orderRoute = createRoute({ getParentRoute: () => rootRoute, path: '/orders/$id', component: Confirmation });
const router = createRouter({ routeTree: rootRoute.addChildren([indexRoute, orderRoute]) });
const queryClient = new QueryClient();

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>
);
