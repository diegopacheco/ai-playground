import { useState } from "react";
import type { AgentLog } from "../types";

interface Props {
  logs: AgentLog[];
}

function LogEntry({ log }: { log: AgentLog }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="log-entry">
      <div className="log-header" onClick={() => setExpanded(!expanded)}>
        <span className="log-toggle">{expanded ? "[-]" : "[+]"}</span>
        <span>
          Log #{log.id} - {log.action_type} - {log.timestamp} - {log.llm_agent}{" "}
          ({log.llm_model})
        </span>
        {log.commit_sha && (
          <span className="log-sha">{log.commit_sha.slice(0, 7)}</span>
        )}
      </div>
      {expanded && (
        <div className="log-body">
          <div className="log-section">
            <div className="log-section-title">PROMPT</div>
            <pre className="log-prompt">{log.prompt}</pre>
          </div>
          <div className="log-section">
            <div className="log-section-title">RESPONSE</div>
            <pre className="log-response">{log.response}</pre>
          </div>
          <div className="log-section">
            <div className="log-section-title">RESULT</div>
            <div className="log-result">{log.result}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AgentLogs({ logs }: Props) {
  const sorted = [...logs].sort((a, b) => b.id - a.id);

  return (
    <div className="agent-logs">
      <h2>Agent Logs</h2>
      {sorted.length === 0 && (
        <p className="empty-state">No agent logs yet.</p>
      )}
      {sorted.map((log) => (
        <LogEntry key={log.id} log={log} />
      ))}
    </div>
  );
}
