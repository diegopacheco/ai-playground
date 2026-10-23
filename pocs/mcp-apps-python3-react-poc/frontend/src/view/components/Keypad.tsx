import { KEYS, type Key } from "../keypad";

type KeypadProps = {
  onKey: (key: Key) => void;
  disabled: boolean;
};

export function Keypad({ onKey, disabled }: KeypadProps) {
  return (
    <div className="keypad">
      {KEYS.map((key) => (
        <button
          key={key.label}
          type="button"
          className={`key key-${key.kind}${key.wide ? " key-wide" : ""}`}
          onClick={() => onKey(key)}
          disabled={disabled && key.kind === "equals"}
          aria-label={key.label === "⌫" ? "backspace" : key.label}
        >
          {key.label}
        </button>
      ))}
    </div>
  );
}
