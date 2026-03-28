import type { Counters, TestClassification } from "../types";

interface Props {
  counters: Counters | null;
  testClassification: TestClassification | null;
}

export default function CounterCards({ counters, testClassification }: Props) {
  const cards = [
    { label: "Total Cycles", value: counters?.total_cycles ?? 0 },
    { label: "Compilation Fixes", value: counters?.compilation_fixes ?? 0 },
    { label: "Test Fixes", value: counters?.test_fixes ?? 0 },
    { label: "Tests Added", value: counters?.tests_added ?? 0 },
    { label: "Comments Answered", value: counters?.comments_answered ?? 0 },
  ];

  const tc = testClassification;
  const hasTests = tc && (tc.unit + tc.integration + tc.e2e + tc.other) > 0;

  return (
    <div>
      <div className="counter-cards">
        {cards.map((card) => (
          <div key={card.label} className="counter-card">
            <div className="counter-number">{card.value}</div>
            <div className="counter-label">{card.label}</div>
          </div>
        ))}
      </div>
      {hasTests && (
        <div className="test-classification">
          <span className="test-class-title">Tests by type:</span>
          <span className="test-class-badge unit">unit: {tc.unit}</span>
          <span className="test-class-badge integration">integration: {tc.integration}</span>
          <span className="test-class-badge e2e">e2e: {tc.e2e}</span>
          {tc.other > 0 && <span className="test-class-badge other">other: {tc.other}</span>}
        </div>
      )}
    </div>
  );
}
