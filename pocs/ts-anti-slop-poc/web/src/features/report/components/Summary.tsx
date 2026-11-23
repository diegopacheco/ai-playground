import type { Report } from "../types.ts";

type SummaryProps = { readonly report: Report };

export function Summary({ report }: SummaryProps) {
  const total = report.samples.length;
  const rate = Math.round((report.caught / total) * 100);

  return (
    <section className="summary">
      <div className="stat"><span className="num">{total}</span><span>slop samples</span></div>
      <div className="stat good"><span className="num">{report.caught}</span><span>caught</span></div>
      <div className="stat bad"><span className="num">{report.missed}</span><span>missed</span></div>
      <div className="stat"><span className="num">{rate}%</span><span>detection rate</span></div>
      <div className={report.surprises === 0 ? "stat" : "stat bad"}><span className="num">{report.surprises}</span><span>surprises vs catalog</span></div>
    </section>
  );
}
