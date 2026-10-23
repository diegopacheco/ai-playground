import type { Calculation } from "../../shared/calculation";

type HistoryProps = {
  items: Calculation[];
  onPick: (expression: string) => void;
};

export function History({ items, onPick }: HistoryProps) {
  return (
    <section className="panel">
      <h2>History</h2>
      {items.length === 0 ? (
        <p className="empty">No calculations yet.</p>
      ) : (
        <ul className="history">
          {items.map((item, index) => (
            <li key={`${index}-${item.expression}`}>
              <button type="button" onClick={() => onPick(item.expression)}>
                <span>{item.expression}</span>
                <strong>= {item.result}</strong>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
