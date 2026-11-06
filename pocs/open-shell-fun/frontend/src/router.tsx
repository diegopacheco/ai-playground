import { createRootRoute, createRoute, createRouter, Link, Outlet } from "@tanstack/react-router";
import { AttacksTab, LogTab, PolicyTab, SandboxLayout, TerminalTab } from "./routes/SandboxPage";
import { SandboxesPage } from "./routes/SandboxesPage";

const rootRoute = createRootRoute({
  component: () => (
    <div className="app">
      <header className="top">
        <Link to="/" className="brand">
          <span className="logo">&gt;_</span> OpenShell <span className="light">Policy Lab</span>
        </Link>
        <span className="muted">safe, private runtime for autonomous agents</span>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  ),
});

const indexRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: SandboxesPage });

const sandboxRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/sandboxes/$name",
  component: function Layout() {
    return <SandboxLayout name={sandboxRoute.useParams().name} />;
  },
});

const useName = () => sandboxRoute.useParams().name;

const terminalRoute = createRoute({ getParentRoute: () => sandboxRoute, path: "/", component: () => <TerminalTab name={useName()} /> });
const attacksRoute = createRoute({ getParentRoute: () => sandboxRoute, path: "attacks", component: () => <AttacksTab name={useName()} /> });
const policyRoute = createRoute({ getParentRoute: () => sandboxRoute, path: "policy", component: () => <PolicyTab name={useName()} /> });
const logRoute = createRoute({ getParentRoute: () => sandboxRoute, path: "log", component: () => <LogTab name={useName()} /> });

const routeTree = rootRoute.addChildren([
  indexRoute,
  sandboxRoute.addChildren([terminalRoute, attacksRoute, policyRoute, logRoute]),
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
