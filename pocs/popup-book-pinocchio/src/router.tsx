import { createHashHistory, createRootRoute, createRoute, createRouter, redirect } from "@tanstack/react-router";
import { parsePageParam } from "./story/navigation";
import { PAGES } from "./story/pages";
import { TheaterView } from "./ui/TheaterView";

const rootRoute = createRootRoute();

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  beforeLoad: () => {
    throw redirect({ to: "/page/$page", params: { page: "1" } });
  },
});

const pageRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/page/$page",
  component: PageScreen,
});

function PageScreen() {
  const { page } = pageRoute.useParams();
  return <TheaterView page={parsePageParam(page, PAGES.length)} />;
}

export const router = createRouter({ routeTree: rootRoute.addChildren([indexRoute, pageRoute]), history: createHashHistory() });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
