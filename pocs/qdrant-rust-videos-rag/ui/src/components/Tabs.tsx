import type { TabName } from "../types";

const TABS: { name: TabName; label: string }[] = [
  { name: "ask", label: "Ask" },
  { name: "search", label: "Search" },
  { name: "similar", label: "Similar" },
];

type Props = {
  active: TabName;
  onChange: (tab: TabName) => void;
};

export function Tabs({ active, onChange }: Props) {
  return (
    <nav className="tabs">
      {TABS.map((tab) => (
        <button
          key={tab.name}
          className={tab.name === active ? "tab active" : "tab"}
          onClick={() => onChange(tab.name)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
