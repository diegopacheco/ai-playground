import type { ConversationEntry, AgentAction } from "../types";

interface Props {
  entries: ConversationEntry[];
  actions: AgentAction[];
}

export default function PrConversation({ entries, actions }: Props) {
  if (entries.length === 0 && actions.length === 0) {
    return <div className="empty-state">No conversation yet.</div>;
  }

  return (
    <div className="conversation">
      <h2>PR Conversation</h2>
      {entries.map((entry, i) => (
        <div
          key={`conv-${entry.id}-${i}`}
          className={`conversation-entry conversation-${entry.type}`}
        >
          <div className="conversation-meta">
            <span className="conversation-author">{entry.author}</span>
            <span className={`conversation-badge ${entry.type}`}>
              {entry.type}
            </span>
            {entry.created_at && (
              <span className="conversation-time">{entry.created_at}</span>
            )}
          </div>
          {entry.file_path && (
            <div className="conversation-file">
              {entry.file_path}
              {entry.line ? `:${entry.line}` : ""}
            </div>
          )}
          <div className="conversation-body">{entry.body}</div>
        </div>
      ))}
      {actions.length > 0 && (
        <>
          <h3>Agent Actions</h3>
          {actions.map((action) => (
            <div
              key={`action-${action.id}`}
              className="conversation-entry conversation-action"
            >
              <div className="conversation-meta">
                <span className="conversation-author">{action.llm_agent} ({action.llm_model})</span>
                <span className="conversation-badge action">{action.action_type}</span>
                <span className="conversation-time">{action.timestamp}</span>
              </div>
              {action.files_changed.length > 0 && (
                <div className="conversation-file">
                  {action.files_changed.join(", ")}
                </div>
              )}
              <div className="conversation-body">{action.description}</div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
