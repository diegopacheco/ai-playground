import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Layout } from "./components/Layout";
import { Agents } from "./routes/Agents";
import { Annotate } from "./routes/Annotate";
import { Chat } from "./routes/Chat";
import { Ingest } from "./routes/Ingest";
import { Rust } from "./routes/Rust";
import { Search } from "./routes/Search";
import "./styles.css";

const rootRoute = createRootRoute({ component: Layout });

const routes = [
  createRoute({ getParentRoute: () => rootRoute, path: "/", component: Ingest }),
  createRoute({ getParentRoute: () => rootRoute, path: "/chat", component: Chat }),
  createRoute({ getParentRoute: () => rootRoute, path: "/rust", component: Rust }),
  createRoute({ getParentRoute: () => rootRoute, path: "/search", component: Search }),
  createRoute({ getParentRoute: () => rootRoute, path: "/agents", component: Agents }),
  createRoute({ getParentRoute: () => rootRoute, path: "/annotate", component: Annotate }),
];

const router = createRouter({ routeTree: rootRoute.addChildren(routes) });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
});

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
