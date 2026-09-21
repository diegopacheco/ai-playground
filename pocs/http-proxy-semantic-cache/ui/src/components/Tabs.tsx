export interface Tab<K extends string> {
  key: K;
  label: string;
}

interface TabsProps<K extends string> {
  tabs: Tab<K>[];
  active: K;
  onChange: (key: K) => void;
}

export function Tabs<K extends string>({ tabs, active, onChange }: TabsProps<K>) {
  return (
    <nav className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button key={tab.key} role="tab" aria-selected={tab.key === active} className={tab.key === active ? "tab active" : "tab"} onClick={() => onChange(tab.key)}>
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
