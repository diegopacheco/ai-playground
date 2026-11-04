import { useQuery } from "@tanstack/react-query";
import { fetchEvaluators } from "../api";

export function Evaluators() {
  const evaluators = useQuery({ queryKey: ["evaluators"], queryFn: fetchEvaluators });
  if (evaluators.isPending) return <p className="muted">Loading evaluators</p>;
  if (evaluators.isError) return <div className="alert">{evaluators.error.message}</div>;
  return (
    <section className="panel">
      <h2>Common evaluators</h2>
      <p className="muted">
        Every check in every suite is graded on the evals service by one of these, so scores from skills and MCP servers are
        comparable.
      </p>
      <div className="evaluators">
        {evaluators.data.map((e) => (
          <div key={e.name} className="evaluator">
            <code>{e.name}</code>
            <p>{e.description}</p>
            <span className="muted mono">value: {e.value}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
