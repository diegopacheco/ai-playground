import { useState } from "react";
import { Icon } from "./icons";
import { Modal } from "./panels";
import { filterShortcuts, keysOf, SHORTCUTS } from "./shortcuts";

export function ShortcutsModal({ onClose }: { readonly onClose: () => void }) {
  const [query, setQuery] = useState("");
  const groups = filterShortcuts(query);
  const count = groups.reduce((n, g) => n + g.shortcuts.length, 0);

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape" && query !== "") {
      e.preventDefault();
      e.nativeEvent.stopImmediatePropagation();
      setQuery("");
    }
  };

  return (
    <Modal id="shortcuts" title="Keyboard shortcuts" icon={<Icon name="keyboard" />} onClose={onClose} className="shortcuts-modal">
      <div className="search-row">
        <Icon name="search" />
        <input autoFocus value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={onKeyDown} placeholder="Filter shortcuts" aria-label="Filter shortcuts" />
        <span className="muted">
          {count} of {SHORTCUTS.length}
        </span>
      </div>
      {groups.length === 0 && <p className="empty">No shortcut matches "{query}".</p>}
      <div className="shortcut-groups">
        {groups.map(({ group, shortcuts }) => (
          <section key={group.id} className="shortcut-group" style={{ borderColor: group.color, color: group.color }}>
            <h3>
              <Icon name={group.icon} />
              {group.title}
            </h3>
            <ul>
              {shortcuts.map((s) => (
                <li key={s.id}>
                  <Icon name={s.icon} />
                  <span className="shortcut-label">{s.label}</span>
                  <span className="keys">
                    {keysOf(s.combo).map((k) => (
                      <kbd key={k} style={{ borderColor: group.color, color: group.color }}>
                        {k}
                      </kbd>
                    ))}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </Modal>
  );
}
