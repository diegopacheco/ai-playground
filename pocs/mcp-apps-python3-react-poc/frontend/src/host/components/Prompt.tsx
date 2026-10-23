import { useState, type FormEvent } from "react";

const SUGGESTIONS = ["(2 + 3) * 4 ^ 2", "1 / 3 + 1 / 6", "-(7 - 10) % 2", "0.1 + 0.2"];

type PromptProps = {
  disabled: boolean;
  onSubmit: (expression: string) => void;
};

export function Prompt({ disabled, onSubmit }: PromptProps) {
  const [value, setValue] = useState("");

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (value.trim() === "") return;
    onSubmit(value.trim());
    setValue("");
  };

  return (
    <div className="prompt">
      <form onSubmit={submit}>
        <input
          aria-label="Expression"
          placeholder="Ask the model to calculate, e.g. (2 + 3) * 4"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          disabled={disabled}
        />
        <button type="submit" disabled={disabled}>
          Call tool
        </button>
      </form>
      <div className="chips">
        {SUGGESTIONS.map((suggestion) => (
          <button key={suggestion} type="button" disabled={disabled} onClick={() => onSubmit(suggestion)}>
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
