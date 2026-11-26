type PresetsProps = { readonly items: readonly string[]; readonly onPick: (text: string) => void };

export function Presets({ items, onPick }: PresetsProps) {
  return (
    <div className="presets">
      <span className="presets-title">Try</span>
      {items.map((text) => (
        <button key={text} type="button" className="preset" onClick={() => onPick(text)}>
          {text}
        </button>
      ))}
    </div>
  );
}
