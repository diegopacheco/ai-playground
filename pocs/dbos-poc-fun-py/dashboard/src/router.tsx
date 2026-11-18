import { Link, Outlet, createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { WorkflowPage } from "./pages/WorkflowPage.tsx";
import { WorkflowsPage } from "./pages/WorkflowsPage.tsx";

const rootRoute = createRootRoute({
  component: () => (
    <>
      <header className="nav">
        <Link to="/" className="brand">DBOS Dashboard</Link>
        <span className="nav-tag">workflows, steps and events read from the DBOS system tables in Postgres</span>
      </header>
      <div className="shell"><Outlet /></div>
    </>
  ),
});

const workflowsRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: WorkflowsPage });

export const workflowRoute = createRoute({ getParentRoute: () => rootRoute, path: "/workflows/$id", component: WorkflowPage });

export const router = createRouter({ routeTree: rootRoute.addChildren([workflowsRoute, workflowRoute]) });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
