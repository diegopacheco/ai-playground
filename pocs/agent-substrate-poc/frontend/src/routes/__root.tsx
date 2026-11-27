import { Link, Outlet, createRootRoute } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: RootLayout,
  errorComponent: ({ error }) => <p className="error page">{error instanceof Error ? error.message : String(error)}</p>,
});

function RootLayout() {
  return (
    <div className="app">
      <header className="top">
        <Link to="/" className="brand">
          <img src="/logo.svg" alt="" width={32} height={32} />
          Agent Substrate POC
        </Link>
        <nav>
          <Link to="/" activeProps={{ className: "active" }} activeOptions={{ exact: true }}>
            Actors
          </Link>
          <Link to="/workers" activeProps={{ className: "active" }}>
            Workers
          </Link>
        </nav>
      </header>
      <main className="page">
        <Outlet />
      </main>
    </div>
  );
}
