export type Tone = "good" | "warn" | "bad" | "info";

type PillProps = { readonly text: string; readonly tone: Tone };

export function Pill({ text, tone }: PillProps) {
  return <span className={`pill pill-${tone}`}>{text}</span>;
}
