import { useEffect, useState } from "react";
import { keysFor } from "./shortcuts";

const DELAY_MS = 400;
const MARGIN = 8;

type Tip = { readonly label: string; readonly keys: string; readonly x: number; readonly y: number; readonly below: boolean };

function place(el: Element, label: string, keys: string): Tip {
  const r = el.getBoundingClientRect();
  const width = (label.length + keys.length) * 7.5 + 40;
  const x = Math.min(Math.max(r.left + r.width / 2 - width / 2, MARGIN), window.innerWidth - width - MARGIN);
  const below = r.bottom + 40 < window.innerHeight;
  return { label, keys, x, y: below ? r.bottom + 6 : r.top - 34, below };
}

export function Tooltip() {
  const [tip, setTip] = useState<Tip | undefined>(undefined);

  useEffect(() => {
    let timer = 0;
    let current: Element | null = null;
    const clear = () => {
      window.clearTimeout(timer);
      current = null;
      setTip(undefined);
    };
    const onOver = (e: MouseEvent) => {
      if (!(e.target instanceof Element)) return;
      const el = e.target.closest("[data-tip]");
      if (el === current) return;
      clear();
      if (!el) return;
      current = el;
      timer = window.setTimeout(() => {
        const label = el.getAttribute("data-tip") ?? "";
        const keys = keysFor(el.getAttribute("data-keys") ?? "");
        setTip(place(el, label, keys));
      }, DELAY_MS);
    };
    document.addEventListener("mouseover", onOver);
    document.addEventListener("mousedown", clear);
    window.addEventListener("blur", clear);
    return () => {
      document.removeEventListener("mouseover", onOver);
      document.removeEventListener("mousedown", clear);
      window.removeEventListener("blur", clear);
      window.clearTimeout(timer);
    };
  }, []);

  if (!tip) return null;
  return (
    <div className="tooltip" role="tooltip" style={{ left: tip.x, top: tip.y }}>
      <span>{tip.label}</span>
      {tip.keys && <kbd>{tip.keys}</kbd>}
    </div>
  );
}
