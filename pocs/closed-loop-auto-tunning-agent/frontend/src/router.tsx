import { createRootRoute, createRoute, createRouter, Link, Outlet } from "@tanstack/react-router";
import { CircuitBreakerPage } from "./pages/CircuitBreakerPage";
import { SimplePatternPage } from "./pages/SimplePatternPage";

function Root() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          Resilience4j Auto-Tuning Control Panel
        </div>
        <nav className="tabs">
          <Link to="/" className="tab" activeProps={{ className: "tab active" }} activeOptions={{ exact: true }}>
            Circuit Breaker
          </Link>
          <Link to="/retry" className="tab" activeProps={{ className: "tab active" }}>
            Retry
          </Link>
          <Link to="/ratelimiter" className="tab" activeProps={{ className: "tab active" }}>
            Rate Limiter
          </Link>
          <Link to="/bulkhead" className="tab" activeProps={{ className: "tab active" }}>
            Bulkhead
          </Link>
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  );
}

const rootRoute = createRootRoute({ component: Root });

const cbRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: CircuitBreakerPage });

const retryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/retry",
  component: () => (
    <SimplePatternPage
      title="Retry"
      endpoint="/retry/call"
      metricsKey="retry"
      description="Each call retries up to 3 times with backoff before failing. Raise the failure rate to see retried-then-failed calls climb."
      defaults={{ failRate: 0.7, latencyMs: 40, jitterMs: 20, total: 30, rps: 10 }}
    />
  ),
});

const rateRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/ratelimiter",
  component: () => (
    <SimplePatternPage
      title="Rate Limiter"
      endpoint="/ratelimiter/call"
      metricsKey="ratelimiter"
      description="Allows 5 calls per second. Push the request rate above that and excess calls are rate-limited (rejected)."
      defaults={{ failRate: 0.0, latencyMs: 0, jitterMs: 0, total: 40, rps: 30 }}
    />
  ),
});

const bulkheadRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/bulkhead",
  component: () => (
    <SimplePatternPage
      title="Bulkhead"
      endpoint="/bulkhead/call"
      metricsKey="bulkhead"
      description="Allows 5 concurrent calls. Combine latency with a high request rate to exceed concurrency and see calls rejected."
      defaults={{ failRate: 0.0, latencyMs: 250, jitterMs: 50, total: 40, rps: 30 }}
    />
  ),
});

const routeTree = rootRoute.addChildren([cbRoute, retryRoute, rateRoute, bulkheadRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
