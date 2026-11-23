import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRootRoute, type ErrorComponentProps, Link, Outlet } from "@tanstack/react-router";

const queryClient = new QueryClient();

function RootLayout() {
  return (
    <QueryClientProvider client={queryClient}>
      <header className="top">
        <h1>anti-slop <span>caught vs missed</span></h1>
        <nav>
          <Link to="/" activeOptions={{ exact: true }}>Overview</Link>
          <Link to="/caught">Caught</Link>
          <Link to="/missed">Missed</Link>
        </nav>
      </header>
      <main><Outlet /></main>
    </QueryClientProvider>
  );
}

function RootError({ error }: ErrorComponentProps) {
  return <p className="error">Could not load the report: {error instanceof Error ? error.message : String(error)}</p>;
}

export const Route = createRootRoute({ component: RootLayout, errorComponent: RootError });
