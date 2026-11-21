import { createRootRoute } from "@tanstack/react-router";
import { AppShell } from "../shell/AppShell";
import { parseWeekSearch } from "../shell/week";

export const Route = createRootRoute({
  validateSearch: parseWeekSearch,
  component: AppShell,
  errorComponent: ({ error }) => (
    <div className="route-error" role="alert">
      <h2>Something went wrong</h2>
      <p>{error instanceof Error ? error.message : String(error)}</p>
    </div>
  ),
});
