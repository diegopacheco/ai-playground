import { Link, Outlet, createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { AgentsPage } from "./pages/AgentsPage";
import { LivePage } from "./pages/LivePage";
import { PollPage } from "./pages/PollPage";
import { TypesPage } from "./pages/TypesPage";

const tabs = [
  { to: "/", label: "Live (push)" },
  { to: "/poll", label: "Poll" },
  { to: "/types", label: "Event types" },
  { to: "/agents", label: "Claude Code / Codex" },
] as const;

function Layout() {
  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="logo">⚡</span>
          <div>
            <h1>MCP Events</h1>
            <p className="dim">draft events/* extension · poll + push · Python 3.15 server</p>
          </div>
        </div>
        <nav>
          {tabs.map((tab) => (
            <Link key={tab.to} to={tab.to} className="tab" activeProps={{ className: "tab active" }} activeOptions={{ exact: true }}>
              {tab.label}
            </Link>
          ))}
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  );
}

const root = createRootRoute({ component: Layout });

const routeTree = root.addChildren([
  createRoute({ getParentRoute: () => root, path: "/", component: LivePage }),
  createRoute({ getParentRoute: () => root, path: "/poll", component: PollPage }),
  createRoute({ getParentRoute: () => root, path: "/types", component: TypesPage }),
  createRoute({ getParentRoute: () => root, path: "/agents", component: AgentsPage }),
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
