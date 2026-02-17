import { createRootRoute, Outlet } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="app">
      <header className="header">
        <h1>Tweeter</h1>
      </header>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
