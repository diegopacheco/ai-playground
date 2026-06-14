import { metrics } from "../data.js";

export default function MetricsTable() {
  const ranked = [...metrics]
    .map((m) => ({
      ...m,
      score: m.revenue * m.margin - m.refunds * 12.5,
      perUnit: m.units > 0 ? m.revenue / m.units : 0,
    }))
    .sort((a, b) => b.score - a.score);

  const totals = ranked.reduce(
    (acc, m) => {
      acc.revenue += m.revenue;
      acc.units += m.units;
      acc.refunds += m.refunds;
      return acc;
    },
    { revenue: 0, units: 0, refunds: 0 }
  );

  const top = ranked.slice(0, 40);

  return (
    <section className="metrics-table">
      <h2 className="metrics-table__title">SKU Performance</h2>
      <div className="metrics-table__totals">
        <span>Revenue ${totals.revenue.toFixed(0)}</span>
        <span>Units {totals.units}</span>
        <span>Refunds {totals.refunds}</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>SKU</th>
            <th>Region</th>
            <th>Category</th>
            <th className="num">Units</th>
            <th className="num">Revenue</th>
            <th className="num">Margin</th>
            <th className="num">Score</th>
          </tr>
        </thead>
        <tbody>
          {top.map((m) => (
            <tr key={m.id}>
              <td>{m.sku}</td>
              <td>{m.region}</td>
              <td>{m.category}</td>
              <td className="num">{m.units}</td>
              <td className="num">${m.revenue.toFixed(0)}</td>
              <td className="num">{(m.margin * 100).toFixed(1)}%</td>
              <td className="num">{m.score.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
