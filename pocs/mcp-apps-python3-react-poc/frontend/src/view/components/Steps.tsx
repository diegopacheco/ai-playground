import { prettyStep, type Calculation } from "../../shared/calculation";

type StepsProps = {
  calculation: Calculation | null;
};

export function Steps({ calculation }: StepsProps) {
  return (
    <section className="panel">
      <h2>Evaluation steps</h2>
      {calculation && calculation.steps.length > 0 ? (
        <ol className="steps">
          {calculation.steps.map((step, index) => (
            <li key={`${index}-${step}`} style={{ animationDelay: `${index * 90}ms` }}>
              {prettyStep(step)}
            </li>
          ))}
        </ol>
      ) : (
        <p className="empty">Press = to see how the server evaluates the expression.</p>
      )}
    </section>
  );
}
