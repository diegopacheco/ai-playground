import { Link, Outlet, createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { ChatPage } from "./pages/ChatPage";
import { DiscoverPage } from "./pages/DiscoverPage";
import { FeedbackPage } from "./pages/FeedbackPage";
import { InspectorPage } from "./pages/InspectorPage";
import { RoomsPage } from "./pages/RoomsPage";
import { WirePage } from "./pages/WirePage";

const tabs = [
  { to: "/", label: "Discover" },
  { to: "/rooms", label: "Rooms API" },
  { to: "/chat", label: "Company Agent" },
  { to: "/feedback", label: "Feedback" },
  { to: "/inspector", label: "Inspector" },
  { to: "/wire", label: "Wire log" },
] as const;

function Layout() {
  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <img src="/logo.svg" alt="" className="logo" />
          <div>
            <h1>Atlas, a Poppy Personal Agent</h1>
            <p className="dim">Personal Agent Protocol draft 0.4 against Willow Inn</p>
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
  createRoute({ getParentRoute: () => root, path: "/", component: DiscoverPage }),
  createRoute({ getParentRoute: () => root, path: "/rooms", component: RoomsPage }),
  createRoute({ getParentRoute: () => root, path: "/chat", component: ChatPage }),
  createRoute({ getParentRoute: () => root, path: "/feedback", component: FeedbackPage }),
  createRoute({ getParentRoute: () => root, path: "/inspector", component: InspectorPage }),
  createRoute({ getParentRoute: () => root, path: "/wire", component: WirePage }),
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
