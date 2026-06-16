import { createRouter, createRoute, createRootRoute } from "@tanstack/react-router";
import { RootLayout } from "./routes/root";
import { Home } from "./routes/home";
import { About } from "./routes/about";

const rootRoute = createRootRoute({ component: RootLayout });
const homeRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: Home });
const aboutRoute = createRoute({ getParentRoute: () => rootRoute, path: "/about", component: About });

const routeTree = rootRoute.addChildren([homeRoute, aboutRoute]);

export const router = createRouter({ routeTree });
