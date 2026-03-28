import type { PrInfo } from "../types";

interface Props {
  prInfo: PrInfo | null;
  activeTab: string;
  onTabChange: (tab: string) => void;
  onRunAgent: () => void;
  running: boolean;
}

const TAB_LABELS: Record<string, string> = {
  dashboard: "Dashboard",
  conversation: "Conversation",
  logs: "Agent Logs",
};

export default function Header({ prInfo, activeTab, onTabChange, onRunAgent, running }: Props) {
  return (
    <div className="header">
      <div className="header-top">
        <h1 className="header-title">Agent PR Monitor</h1>
        {prInfo && (
          <div className="header-info">
            <a
              href={prInfo.url}
              target="_blank"
              rel="noopener noreferrer"
              className="pr-link"
            >
              {prInfo.owner}/{prInfo.repo}#{prInfo.pr_number} - {prInfo.title}
            </a>
            <span className="header-meta">
              branch: <strong>{prInfo.branch}</strong> | files:{" "}
              <strong>{prInfo.total_files}</strong> | agent:{" "}
              <strong>{prInfo.agent_name}</strong> ({prInfo.agent_model})
            </span>
          </div>
        )}
      </div>
      <div className="header-bottom">
        <div className="tabs">
          {Object.keys(TAB_LABELS).map((tab) => (
            <button
              key={tab}
              className={`tab-btn ${activeTab === tab ? "active" : ""}`}
              onClick={() => onTabChange(tab)}
            >
              {TAB_LABELS[tab]}
            </button>
          ))}
        </div>
        <button
          className="run-agent-btn"
          onClick={onRunAgent}
          disabled={running}
        >
          {running ? "Running..." : "Run Agent"}
        </button>
      </div>
    </div>
  );
}
