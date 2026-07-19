import { Badge } from "@design/Badge/Badge";
import "./ReadOnlyNotice.css";

export interface ReadOnlyNoticeProps {
  message: string;
  readOnlyViolation: boolean;
}

export function ReadOnlyNotice({ message, readOnlyViolation }: ReadOnlyNoticeProps) {
  return (
    <div className={`console-notice ${readOnlyViolation ? "console-notice-denied" : "console-notice-error"}`} role="alert">
      <Badge tone="error">{readOnlyViolation ? "read only" : "error"}</Badge>
      <span className="console-notice-text">{message}</span>
      {readOnlyViolation ? (
        <span className="console-notice-hint">
          This console never writes. Run the statement in a tool that is allowed to change data.
        </span>
      ) : null}
    </div>
  );
}
