import { createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { App } from "./App";
import { ReposTab } from "./tabs/ReposTab";
import { DashboardTab } from "./tabs/DashboardTab";
import { IssuesTab } from "./tabs/IssuesTab";
import { ActionCenterTab } from "./tabs/ActionCenterTab";
import { InsightsTab } from "./tabs/InsightsTab";
import { SettingsTab } from "./tabs/SettingsTab";

const rootRoute = createRootRoute({ component: App });

const reposRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: ReposTab });
const dashboardRoute = createRoute({ getParentRoute: () => rootRoute, path: "/dashboard", component: DashboardTab });
const issuesRoute = createRoute({ getParentRoute: () => rootRoute, path: "/issues", component: IssuesTab });
const actionRoute = createRoute({ getParentRoute: () => rootRoute, path: "/action-center", component: ActionCenterTab });
const insightsRoute = createRoute({ getParentRoute: () => rootRoute, path: "/insights", component: InsightsTab });
const settingsRoute = createRoute({ getParentRoute: () => rootRoute, path: "/settings", component: SettingsTab });

const routeTree = rootRoute.addChildren([
  reposRoute,
  dashboardRoute,
  issuesRoute,
  actionRoute,
  insightsRoute,
  settingsRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
