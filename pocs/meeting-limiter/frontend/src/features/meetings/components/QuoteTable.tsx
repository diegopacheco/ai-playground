import { PersonCell } from "../../../components/Person";
import { formatMinutes, formatMoney } from "../../../money";
import type { Quote } from "../types";

export function QuoteTable({ quote }: { readonly quote: Quote }) {
  const short = quote.lines.filter((l) => !l.allowed);
  return (
    <div className="quote">
      <div className={`quote-total ${quote.allowed ? "ok" : "blocked"}`}>
        <span>
          {formatMinutes(quote.minutes)} x {quote.lines.length} {quote.lines.length === 1 ? "person" : "people"}
        </span>
        <strong>{formatMoney(quote.totalCost)}</strong>
      </div>
      {!quote.allowed && (
        <p className="quote-warning" role="alert">
          Not enough funds: {short.map((l) => `${l.user.name} is short ${formatMoney(-l.balanceAfter)}`).join(", ")}. Wallets refill next Monday.
        </p>
      )}
      <table className="data-table compact">
        <thead>
          <tr>
            <th>Person</th>
            <th>Cost</th>
            <th>Balance</th>
            <th>After</th>
          </tr>
        </thead>
        <tbody>
          {quote.lines.map((l) => (
            <tr key={l.user.id} className={l.allowed ? undefined : "short"}>
              <td>
                <PersonCell user={l.user} />
              </td>
              <td>{formatMoney(l.cost)}</td>
              <td>{formatMoney(l.balanceBefore)}</td>
              <td className={l.allowed ? "money-ok" : "money-bad"}>{formatMoney(l.balanceAfter)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
