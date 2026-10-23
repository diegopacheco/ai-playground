import { useEffect } from "react";
import { keyForInput, press, type Key } from "./keypad";
import { useCalculator } from "./useCalculator";
import { Display } from "./components/Display";
import { Keypad } from "./components/Keypad";
import { Steps } from "./components/Steps";
import { History } from "./components/History";

export function CalculatorApp() {
  const calculator = useCalculator();
  const { expression, setExpression, evaluate } = calculator;

  const onKey = (key: Key) => {
    if (key.kind === "equals") {
      void evaluate();
      return;
    }
    setExpression(press(expression, key));
  };

  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      const key = keyForInput(event.key);
      if (!key) return;
      event.preventDefault();
      if (key.kind === "equals") void evaluate();
      else setExpression(press(expression, key));
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [expression, setExpression, evaluate]);

  return (
    <main className="calculator">
      <section className="device" aria-label="Calculator">
        <header className="brand">
          <span className={calculator.connected ? "led on" : "led"} aria-hidden="true" />
          <span>MCP·CALC</span>
          <span className="status">{calculator.connected ? "connected" : "connecting"}</span>
        </header>
        <Display
          expression={expression}
          previous={calculator.current?.expression ?? ""}
          error={calculator.error}
          busy={calculator.busy}
        />
        <Keypad onKey={onKey} disabled={calculator.busy} />
      </section>
      <aside className="tape">
        <Steps calculation={calculator.current} />
        <History items={calculator.history} onPick={setExpression} />
      </aside>
    </main>
  );
}
