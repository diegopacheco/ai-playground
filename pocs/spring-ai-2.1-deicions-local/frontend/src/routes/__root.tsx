import { Link, Outlet, createRootRoute } from "@tanstack/react-router";

export const Route = createRootRoute({ component: RootLayout });

const TABS = [
  { to: "/triage", label: "Triage" },
  { to: "/guardrail", label: "Guardrail" },
  { to: "/judge", label: "Judge" },
] as const;

function RootLayout() {
  return (
    <div className="shell">
      <header className="top">
        <div>
          <h1>Decision Desk</h1>
          <p>Spring AI 2.1 DecisionModel on a local Ollama model. Yes, no, and how sure.</p>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <Link key={t.to} to={t.to} className="tab" activeProps={{ className: "tab tab-on" }}>
              {t.label}
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
