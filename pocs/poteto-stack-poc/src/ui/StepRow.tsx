import { useState } from "react";
import type { Step } from "../domain/run";

type Props = {
  step: Step;
  onDone: (evidence: string) => void;
  onSkip: (reason: string) => void;
  onReopen: () => void;
};

export function StepRow({ step, onDone, onSkip, onReopen }: Props) {
  const [note, setNote] = useState("");
  if (step.state !== "todo") {
    return (
      <div className={`step ${step.state}`}>
        <span className="label">{step.state === "done" ? "done" : "skip:"}</span>
        <div>
          <p>{step.text}</p>
          <p className="muted">{step.state === "done" ? `evidence: ${step.evidence}` : `reason: ${step.reason}`}</p>
        </div>
        <button onClick={onReopen}>Reopen</button>
      </div>
    );
  }
  return (
    <div className="step">
      <span className="label">todo</span>
      <div>
        <p>{step.text}</p>
        <input placeholder="evidence or skip reason" value={note} onChange={(e) => setNote(e.target.value)} />
      </div>
      <div className="actions">
        <button disabled={!note.trim()} onClick={() => onDone(note)}>Done</button>
        <button disabled={!note.trim()} onClick={() => onSkip(note)}>Skip</button>
      </div>
    </div>
  );
}
