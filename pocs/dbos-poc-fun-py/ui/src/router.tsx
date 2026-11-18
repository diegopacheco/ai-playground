import { Link, Outlet, createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { RunPage } from "./pages/RunPage.tsx";
import { RunsPage } from "./pages/RunsPage.tsx";

const rootRoute = createRootRoute({
  component: () => (
    <div className="shell">
      <header className="top">
        <Link to="/" className="brand">DBOS Agent Runs</Link>
        <span className="tag">plan, work, summarize as a durable DBOS workflow on Postgres</span>
      </header>
      <Outlet />
    </div>
  ),
});

const runsRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: RunsPage });

export const runRoute = createRoute({ getParentRoute: () => rootRoute, path: "/runs/$runId", component: RunPage });

export const router = createRouter({ routeTree: rootRoute.addChildren([runsRoute, runRoute]) });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
