import { useState } from "react";
import { Icon, type IconName } from "./icons";
import { Modal } from "./panels";
import { keysFor } from "./shortcuts";

export type Command = {
  readonly id: string;
  readonly label: string;
  readonly hint: string;
  readonly icon: IconName;
  readonly shortcut?: string;
  readonly run: () => void;
};

export function filterCommands(commands: readonly Command[], query: string): readonly Command[] {
  const q = query.trim().toLowerCase();
  if (q === "") return commands;
  return commands.filter((c) => `${c.label} ${c.hint}`.toLowerCase().includes(q));
}

type Props = { readonly commands: readonly Command[]; readonly onClose: () => void };

export function CommandPalette({ commands, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const hits = filterCommands(commands, query);
  const selected = Math.min(active, Math.max(hits.length - 1, 0));

  const run = (c: Command | undefined) => {
    if (!c) return;
    onClose();
    c.run();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((selected + 1) % Math.max(hits.length, 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((selected - 1 + hits.length) % Math.max(hits.length, 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      run(hits[selected]);
    }
  };

  return (
    <Modal id="palette" title="Search" icon={<Icon name="search" />} onClose={onClose} className="palette">
      <div className="search-row">
        <Icon name="search" />
        <input
          autoFocus
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
          }}
          onKeyDown={onKeyDown}
          placeholder="Search tabs, people, meetings and actions"
          aria-label="Search"
        />
      </div>
      <ul className="palette-list" role="listbox">
        {hits.length === 0 && <li className="empty">Nothing matches "{query}".</li>}
        {hits.map((c, i) => (
          <li key={c.id} role="option" aria-selected={i === selected}>
            <button type="button" className={i === selected ? "active" : ""} onMouseEnter={() => setActive(i)} onClick={() => run(c)}>
              <Icon name={c.icon} />
              <span className="palette-label">{c.label}</span>
              <span className="muted">{c.hint}</span>
              {c.shortcut && <kbd>{keysFor(c.shortcut)}</kbd>}
            </button>
          </li>
        ))}
      </ul>
    </Modal>
  );
}
