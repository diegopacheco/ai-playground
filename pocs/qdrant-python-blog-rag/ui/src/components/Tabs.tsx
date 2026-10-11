import type { TabName } from "../types";

type Props = {
  active: TabName;
  onChange: (tab: TabName) => void;
};

const TABS: { name: TabName; title: string }[] = [
  { name: "ask", title: "Ask" },
  { name: "search", title: "Search" },
];

export function Tabs({ active, onChange }: Props) {
  return (
    <nav className="tabs">
      {TABS.map((tab) => (
        <button
          key={tab.name}
          className={tab.name === active ? "tab tab-active" : "tab"}
          onClick={() => onChange(tab.name)}
        >
          {tab.title}
        </button>
      ))}
    </nav>
  );
}
