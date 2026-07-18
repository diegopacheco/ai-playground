import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import "./CommandPalette.css";

export interface Destination {
  label: string;
  href: string;
  hint: string;
  keywords: string;
  icon: string;
}

export const DESTINATIONS: Destination[] = [
  { label: "Consoles", href: "/?pick=1", hint: "pick a connection, then query it", keywords: "console query editor run sql cql redis kafka etcd elasticsearch home engine connection", icon: "▤" },
  { label: "Projects", href: "/projects", hint: "manage projects and connections", keywords: "projects connections add edit credentials hosts config", icon: "◈" },
  { label: "Audit trail", href: "/audit-trail", hint: "every statement anyone ran", keywords: "audit trail history log denied allowed security who", icon: "◷" },
  { label: "Users", href: "/users", hint: "accounts, roles and passwords", keywords: "users accounts roles admin password access", icon: "◍" },
  { label: "AI settings", href: "/settings/ai", hint: "choose the agent CLI and model", keywords: "ai agent cli claude codex agy model settings llm prompt", icon: "✦" },
  { label: "Swagger", href: "/swagger", hint: "the backend API reference", keywords: "swagger openapi api docs reference endpoints", icon: "◎" }
];

export interface CommandPaletteProps {
  destinations?: Destination[];
  onNavigate?: (href: string) => void;
}

function score(destination: Destination, term: string) {
  if (!term) {
    return 0;
  }
  const needle = term.toLowerCase();
  const label = destination.label.toLowerCase();
  if (label.startsWith(needle)) {
    return 3;
  }
  if (label.includes(needle)) {
    return 2;
  }
  if (`${destination.keywords} ${destination.hint}`.toLowerCase().includes(needle)) {
    return 1;
  }
  return -1;
}

export default function CommandPalette({ destinations = DESTINATIONS, onNavigate }: CommandPaletteProps) {
  const [open, setOpen] = useState(false);
  const [term, setTerm] = useState("");
  const [active, setActive] = useState(0);
  const searchRef = useRef<HTMLInputElement>(null);

  const matches = useMemo(() => {
    return destinations
      .map((destination) => ({ destination, rank: score(destination, term) }))
      .filter((entry) => entry.rank >= 0)
      .sort((left, right) => right.rank - left.rank)
      .map((entry) => entry.destination);
  }, [destinations, term]);

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen((current) => !current);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (open) {
      setTerm("");
      setActive(0);
      window.setTimeout(() => searchRef.current?.focus(), 0);
    }
  }, [open]);

  useEffect(() => {
    setActive(0);
  }, [term]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [open]);

  const go = (destination: Destination | undefined) => {
    if (!destination) {
      return;
    }
    setOpen(false);
    if (onNavigate) {
      onNavigate(destination.href);
      return;
    }
    window.location.href = destination.href;
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      go(matches[active]);
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((current) => (matches.length === 0 ? 0 : (current + 1) % matches.length));
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((current) => (matches.length === 0 ? 0 : (current - 1 + matches.length) % matches.length));
    }
  };

  if (!open) {
    return null;
  }

  return (
    <div className="palette-backdrop" onClick={() => setOpen(false)}>
      <div
        className="palette"
        role="dialog"
        aria-modal="true"
        aria-label="Go to page"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="palette-search-row">
          <span className="palette-search-icon">⌘K</span>
          <input
            ref={searchRef}
            className="palette-search"
            value={term}
            onChange={(event) => setTerm(event.target.value)}
            placeholder="go to page…"
            aria-label="go to page"
          />
        </div>

        <ul className="palette-list" role="listbox" aria-label="pages">
          {matches.length === 0 ? (
            <li className="palette-empty">no page matches “{term}”</li>
          ) : (
            matches.map((destination, index) => (
              <li key={destination.href}>
                <button
                  role="option"
                  aria-selected={index === active}
                  className={index === active ? "palette-item palette-item-active" : "palette-item"}
                  onClick={() => go(destination)}
                  onMouseEnter={() => setActive(index)}
                >
                  <span className="palette-item-icon">{destination.icon}</span>
                  <span className="palette-item-label">{destination.label}</span>
                  <span className="palette-item-hint">{destination.hint}</span>
                  <span className="palette-item-href">{destination.href}</span>
                </button>
              </li>
            ))
          )}
        </ul>

        <footer className="palette-footer">↑↓ move · ↵ go · esc close</footer>
      </div>
    </div>
  );
}
