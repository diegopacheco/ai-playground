import type { AgentAction, ActionType } from "../types";

interface Props {
  actions: AgentAction[];
}

function actionColor(type: ActionType): string {
  switch (type) {
    case "CompileFix":
      return "#f85149";
    case "TestFix":
      return "#d29922";
    case "TestAdd":
      return "#3fb950";
    case "CommentReply":
      return "#58a6ff";
    case "MergeConflictFix":
      return "#c9d1d9";
  }
}

export default function ActivityLog({ actions }: Props) {
  const sorted = [...actions].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  return (
    <div className="activity-log">
      <h3>Activity Log</h3>
      <div className="activity-entries">
        {sorted.map((action) => (
          <div
            key={action.id}
            className="activity-entry"
            style={{ color: actionColor(action.action_type) }}
          >
            [{action.timestamp}] {action.action_type}: {action.description}
            {action.commit_sha && ` (${action.commit_sha.slice(0, 7)})`}
          </div>
        ))}
        {sorted.length === 0 && (
          <div className="activity-entry" style={{ color: "#484f58" }}>
            No actions recorded yet.
          </div>
        )}
      </div>
    </div>
  );
}
