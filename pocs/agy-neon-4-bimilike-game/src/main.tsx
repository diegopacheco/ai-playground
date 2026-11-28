import { QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { logger } from "./lib/logger";
import { queryClient } from "./lib/queryClient";
import { router } from "./router";
import "./styles/app.css";

const container = document.getElementById("root");
if (!container) {
  throw new Error("Root element not found");
}

logger.info("app.boot");

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
