import { DeployForm } from "../components/DeployForm";
import { StreamPanel } from "../components/StreamPanel";
import { useEventTypes } from "../mcp/hooks";

export function LivePage() {
  const types = useEventTypes();
  return (
    <div className="page">
      <p className="lead">
        Push delivery: every panel holds one long-lived <span className="mono">events/stream</span> request. The server answers with an SSE
        stream of <span className="mono">notifications/events/*</span>. The client keeps the cursor, so a resumed stream replays only what it
        missed.
      </p>
      <DeployForm />
      {types.isPending && <div className="empty">Loading event types with events/list...</div>}
      {types.error && <div className="empty error-text">{types.error.message}</div>}
      <div className="streams">
        {types.data?.filter((t) => t.delivery.includes("push")).map((t) => <StreamPanel key={t.name} descriptor={t} />)}
      </div>
    </div>
  );
}
