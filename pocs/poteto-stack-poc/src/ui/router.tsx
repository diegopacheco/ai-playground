import { createRootRoute, createRoute, createRouter, Link, Outlet } from "@tanstack/react-router";
import { catalog } from "../domain/catalog";
import { LabPage } from "./LabPage";
import { PlaybooksPage } from "./PlaybooksPage";
import { PrinciplesPage } from "./PrinciplesPage";

function Shell() {
  return (
    <div className="shell">
      <header>
        <h1>pstack Router Lab</h1>
        <p className="muted">
          poteto-mode ideas on live data: {catalog.skills.length} skills, {catalog.playbooks.length} playbooks, {catalog.principles.length} principles from pstack {catalog.version}
        </p>
        <nav>
          <Link to="/" activeOptions={{ exact: true }}>Lab</Link>
          <Link to="/playbooks">Playbooks</Link>
          <Link to="/principles">Principles</Link>
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  );
}

const rootRoute = createRootRoute({ component: Shell });
const routes = [
  createRoute({ getParentRoute: () => rootRoute, path: "/", component: LabPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/playbooks", component: PlaybooksPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/principles", component: PrinciplesPage }),
];

export const router = createRouter({ routeTree: rootRoute.addChildren(routes) });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
