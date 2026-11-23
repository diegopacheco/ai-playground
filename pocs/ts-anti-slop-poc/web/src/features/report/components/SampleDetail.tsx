import type { SampleResult } from "../types.ts";

type SampleDetailProps = { readonly sample: SampleResult };

export function SampleDetail({ sample }: SampleDetailProps) {
  const flaggedLines = new Set(sample.findings.map((f) => f.line));

  return (
    <div className="detail">
      <div className="file">{sample.file}</div>
      <pre className="code">
        {sample.source.split("\n").map((text, i) => (
          <div key={i} className={flaggedLines.has(i + 1) ? "line flagged" : "line"}>
            <span className="ln">{i + 1}</span>{text}
          </div>
        ))}
      </pre>
      {sample.findings.length === 0 ? (
        <p className="nothing">anti-slop reported nothing for this file.</p>
      ) : (
        <ul className="findings">
          {sample.findings.map((f, i) => (
            <li key={i}><code>{f.rule}</code> line {f.line}: {f.message}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
