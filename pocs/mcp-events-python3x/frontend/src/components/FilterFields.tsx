import type { ObjectSchema } from "../mcp/types";

type Props = {
  schema: ObjectSchema;
  values: Record<string, string>;
  onChange: (values: Record<string, string>) => void;
  disabled?: boolean;
};

export function FilterFields({ schema, values, onChange, disabled }: Props) {
  const update = (key: string, value: string) => {
    const next = { ...values };
    if (value) next[key] = value;
    else delete next[key];
    onChange(next);
  };
  return (
    <div className="filters">
      {Object.entries(schema.properties).map(([key, prop]) => (
        <label key={key} className="field" title={prop.description}>
          <span>{key}</span>
          {prop.enum ? (
            <select value={values[key] ?? ""} disabled={disabled} onChange={(e) => update(key, e.target.value)}>
              <option value="">any</option>
              {prop.enum.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          ) : (
            <input value={values[key] ?? ""} placeholder="any" disabled={disabled} onChange={(e) => update(key, e.target.value)} />
          )}
        </label>
      ))}
    </div>
  );
}
