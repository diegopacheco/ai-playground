import { QueryClient, useQueryClient } from "@tanstack/react-query";
import { Link, Outlet, createRootRoute, createRoute, createRouter, useRouter } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { AccessContext } from "./access";
import { Auth } from "./pages/Auth";
import { Logs } from "./pages/Logs";
import { AppFilter } from "./pages/Scope";
import { State } from "./pages/State";
import { UnauthorizedError } from "./api";
import { Evaluators } from "./pages/Evaluators";
import { Login } from "./pages/Login";
import { Overview } from "./pages/Overview";
import { RunDetail } from "./pages/RunDetail";
import { Runs } from "./pages/Runs";
import { clearSession, loadSession } from "./session";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: (count, error) => !(error instanceof UnauthorizedError) && count < 2 } },
});

function Shell() {
  const [session, setSession] = useState(loadSession());
  const client = useQueryClient();
  const router = useRouter();
  const [app, setApp] = useState("");
  const access = useMemo(() => (session ? { session, app, setApp } : null), [session, app]);
  if (!session || !access) return <Login onSignedIn={() => setSession(loadSession())} />;
  return (
    <AccessContext.Provider value={access}>
    <div className="app">
      <header>
        <div className="brand">
          <div className="brand-mark">AI</div>
          <span>Agent Infra Console</span>
        </div>
        <nav>
          <Link to="/" activeOptions={{ exact: true }}>
            Overview
          </Link>
          <Link to="/logs">Logs</Link>
          <Link to="/state">State</Link>
          <Link to="/runs">Evals</Link>
          <Link to="/auth">Auth</Link>
          <Link to="/evaluators">Evaluators</Link>
        </nav>
        <div className="who">
          <AppFilter />
          <span className="mono">{session.clientId}</span>
          <button
            className="ghost"
            onClick={() => {
              clearSession();
              client.clear();
              setSession(null);
              setApp("");
              void router.navigate({ to: "/" });
            }}
          >
            Sign out
          </button>
        </div>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
    </AccessContext.Provider>
  );
}

const rootRoute = createRootRoute({ component: Shell });
const overviewRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: Overview });
const runsRoute = createRoute({ getParentRoute: () => rootRoute, path: "/runs", component: Runs });
const runRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId",
  component: function RunPage() {
    const { runId } = runRoute.useParams();
    return <RunDetail runId={runId} />;
  },
});
const evaluatorsRoute = createRoute({ getParentRoute: () => rootRoute, path: "/evaluators", component: Evaluators });
const logsRoute = createRoute({ getParentRoute: () => rootRoute, path: "/logs", component: Logs });
const stateRoute = createRoute({ getParentRoute: () => rootRoute, path: "/state", component: State });
const authRoute = createRoute({ getParentRoute: () => rootRoute, path: "/auth", component: Auth });

export const router = createRouter({ routeTree: rootRoute.addChildren([overviewRoute, logsRoute, stateRoute, runsRoute, runRoute, authRoute, evaluatorsRoute]) });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
