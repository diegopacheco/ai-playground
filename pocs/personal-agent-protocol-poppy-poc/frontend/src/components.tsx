import type { ReactNode } from "react";
import { useAgent, type Agent } from "./agent";

export function WithAgent({ children }: { children: (agent: Agent) => ReactNode }) {
  const agent = useAgent();
  if (agent.isPending) return <p className="dim">Discovering Willow Inn and publishing agent keys...</p>;
  if (agent.isError) return <p className="error">Company unreachable: {agent.error.message}. Start it with ./scripts/start-all.sh</p>;
  return <>{children(agent.data)}</>;
}

export function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}

export function ErrorLine({ error }: { error: Error | null }) {
  return error ? <p className="error">{error.message}</p> : null;
}
