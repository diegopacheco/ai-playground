import {
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  type ErrorComponentProps,
} from "@tanstack/react-router";
import { logger } from "./lib/logger";
import { FriendsPage } from "./pages/FriendsPage";
import { HousePage } from "./pages/HousePage";

function RouteErrorView({ error }: ErrorComponentProps) {
  const message = error instanceof Error ? error.message : String(error);
  logger.error("route.error", { message });
  return (
    <div className="error-card">
      <h1>Oops! A toy fell over!</h1>
      <p>{message}</p>
      <a href="/" className="toy-big-btn toy-btn-home">
        🏠 Back to House
      </a>
    </div>
  );
}

const rootRoute = createRootRoute({
  component: () => <Outlet />,
  errorComponent: RouteErrorView,
});

const houseRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: HousePage,
});

const friendsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/friends",
  component: FriendsPage,
});

const routeTree = rootRoute.addChildren([houseRoute, friendsRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
