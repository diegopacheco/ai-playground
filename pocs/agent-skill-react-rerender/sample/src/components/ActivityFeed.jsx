import { activities } from "../data.js";

function classify(a) {
  if (a.verb === "refunded" || a.verb === "returned") return "negative";
  if (a.verb === "purchased") return "positive";
  return "neutral";
}

export default function ActivityFeed() {
  const rows = activities
    .map((a) => ({
      ...a,
      when: new Date(a.ts).toLocaleString(),
      tone: classify(a),
      headline: a.actor + " " + a.verb + " " + a.product,
      money: "$" + a.amount.toFixed(2),
    }))
    .sort((x, y) => y.ts - x.ts);

  return (
    <section className="activity-feed">
      <h2 className="activity-feed__title">Live Activity</h2>
      <ul className="activity-feed__list">
        {rows.map((r) => (
          <li key={r.id} className={"activity-row " + r.tone}>
            <span className="activity-row__dot" />
            <span className="activity-row__text">{r.headline}</span>
            <span className="activity-row__meta">
              {r.region} · {r.money}
            </span>
            <span className="activity-row__when">{r.when}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
