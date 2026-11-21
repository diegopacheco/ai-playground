import { createContext, use, useEffect, useRef, useState, type ReactNode } from "react";
import { Icon } from "./icons";
import { dockSide, hidePanel, parseHidden, restorePanel, type HiddenPanel, type Side } from "./dock";
import { readJson, writeJson } from "./storage";

const STORAGE_KEY = "ml.hiddenPanels";
const SLIDE_MS = 200;

type PanelState = {
  readonly expanded: string | undefined;
  readonly hidden: readonly HiddenPanel[];
  readonly mounted: readonly string[];
  readonly setExpanded: (id: string | undefined) => void;
  readonly hide: (p: HiddenPanel) => void;
  readonly restore: (id: string) => void;
  readonly mount: (id: string) => () => void;
};

const PanelContext = createContext<PanelState | undefined>(undefined);

function usePanels(): PanelState {
  const ctx = use(PanelContext);
  if (!ctx) throw new Error("PanelProvider is missing");
  return ctx;
}

export function PanelProvider({ children }: { readonly children: ReactNode }) {
  const [expanded, setExpanded] = useState<string | undefined>(undefined);
  const [hidden, setHidden] = useState<readonly HiddenPanel[]>(() => readJson(STORAGE_KEY, parseHidden, []));
  const [mounted, setMounted] = useState<readonly string[]>([]);

  useEffect(() => writeJson(STORAGE_KEY, hidden), [hidden]);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopImmediatePropagation();
      setExpanded(undefined);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [expanded]);

  const state: PanelState = {
    expanded,
    hidden,
    mounted,
    setExpanded,
    hide: (p) => {
      setExpanded((e) => (e === p.id ? undefined : e));
      setHidden((list) => hidePanel(list, p));
    },
    restore: (id) => setHidden((list) => restorePanel(list, id)),
    mount: (id) => {
      setMounted((list) => [...list, id]);
      return () =>
        setMounted((list) => {
          const i = list.indexOf(id);
          return i < 0 ? list : [...list.slice(0, i), ...list.slice(i + 1)];
        });
    },
  };
  return (
    <PanelContext value={state}>
      {children}
      <DockTabs />
    </PanelContext>
  );
}

type PanelProps = {
  readonly id: string;
  readonly title: string;
  readonly icon?: ReactNode;
  readonly actions?: ReactNode;
  readonly className?: string;
  readonly children: ReactNode;
};

export function Panel({ id, title, icon, actions, className, children }: PanelProps) {
  const panels = usePanels();
  const ref = useRef<HTMLElement>(null);
  const [leaving, setLeaving] = useState<Side | undefined>(undefined);
  const { mount } = panels;

  useEffect(() => mount(id), [id]);

  if (panels.hidden.some((p) => p.id === id)) return null;
  const expanded = panels.expanded === id;

  const onHide = () => {
    const rect = ref.current?.getBoundingClientRect();
    const side = rect ? dockSide(rect, window.innerWidth) : "right";
    setLeaving(side);
    window.setTimeout(() => {
      setLeaving(undefined);
      panels.hide({ id, title, side });
    }, SLIDE_MS);
  };

  const classes = ["panel", className, expanded ? "panel--expanded" : "", leaving ? `panel--leaving-${leaving}` : ""].filter(Boolean).join(" ");
  return (
    <>
      {expanded && <div className="panel-backdrop" onClick={() => panels.setExpanded(undefined)} />}
      <section ref={ref} className={classes} aria-label={title} data-panel={id}>
        <header className="panel-head">
          {icon}
          <h2>{title}</h2>
          <div className="panel-actions">{actions}</div>
          <div className="panel-buttons">
            <button
              type="button"
              className="icon-button"
              data-tip={expanded ? "Collapse panel" : "Expand panel"}
              data-keys={expanded ? "close" : undefined}
              aria-label={expanded ? `Collapse ${title}` : `Expand ${title}`}
              onClick={() => panels.setExpanded(expanded ? undefined : id)}
            >
              <Icon name={expanded ? "collapse" : "expand"} size={16} />
            </button>
            <button type="button" className="icon-button" data-tip="Hide panel" aria-label={`Hide ${title}`} onClick={onHide}>
              <Icon name="hide" size={16} />
            </button>
          </div>
        </header>
        <div className="panel-body">{children}</div>
      </section>
    </>
  );
}

type ModalProps = PanelProps & { readonly onClose: () => void };

export function Modal({ onClose, ...panel }: ModalProps) {
  const panels = usePanels();
  const { restore } = panels;
  useEffect(() => restore(panel.id), [panel.id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !e.defaultPrevented) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (panels.hidden.some((p) => p.id === panel.id)) return <Panel {...panel} />;
  return (
    <div className="modal-overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <Panel {...panel} className={["modal", panel.className].filter(Boolean).join(" ")} />
    </div>
  );
}

function DockTabs() {
  const panels = usePanels();
  const visible = panels.hidden.filter((p) => panels.mounted.includes(p.id));
  const sides: readonly Side[] = ["left", "right"];
  return (
    <>
      {sides.map((side) => {
        const list = visible.filter((p) => p.side === side);
        if (list.length === 0) return null;
        return (
          <div key={side} className={`dock dock--${side}`}>
            {list.map((p) => (
              <button
                key={p.id}
                type="button"
                className="dock-tab"
                data-tip={`Restore ${p.title}`}
                aria-label={`Restore ${p.title}`}
                onClick={() => panels.restore(p.id)}
              >
                <Icon name={side === "left" ? "arrowRight" : "arrowLeft"} size={14} />
                <span className="dock-title">{p.title}</span>
              </button>
            ))}
          </div>
        );
      })}
    </>
  );
}
