import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";

type Status = "connecting" | "connected" | "closed";

const THEME = {
  background: "#14170f",
  foreground: "#e8f0dc",
  cursor: "#76b900",
  green: "#8fd400",
  brightGreen: "#a8e63a",
  selectionBackground: "#3d6b0066",
};

export function WebTerminal({ name }: { name: string }) {
  const host = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<Status>("connecting");
  const [session, setSession] = useState(0);

  useEffect(() => {
    if (!host.current) {
      return;
    }
    const term = new Terminal({ cursorBlink: true, fontSize: 14, fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace", theme: THEME });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(host.current);
    fit.fit();
    term.writeln(`\x1b[32mopenshell sandbox exec ${name} --tty -- bash -l\x1b[0m`);

    const ws = new WebSocket(api.terminalUrl(name));
    ws.binaryType = "arraybuffer";
    const send = (msg: object) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify(msg));
    const resize = () => {
      fit.fit();
      send({ type: "resize", cols: term.cols, rows: term.rows });
    };
    ws.onopen = () => {
      setStatus("connected");
      resize();
      term.focus();
    };
    ws.onmessage = (e) => term.write(typeof e.data === "string" ? e.data : new Uint8Array(e.data as ArrayBuffer));
    ws.onclose = () => {
      setStatus("closed");
      term.writeln("\r\n\x1b[33m[session closed, the sandbox keeps running]\x1b[0m");
    };
    const input = term.onData((data) => send({ type: "input", data }));
    const observer = new ResizeObserver(resize);
    observer.observe(host.current);

    return () => {
      observer.disconnect();
      input.dispose();
      ws.close();
      term.dispose();
    };
  }, [name, session]);

  return (
    <div className="terminal-wrap">
      <div className="terminal-bar">
        <span className={`dot ${status}`} />
        <span>{name}</span>
        <span className="muted">{status}</span>
        {status === "closed" && (
          <button type="button" className="chip" onClick={() => { setStatus("connecting"); setSession((s) => s + 1); }}>
            Reconnect
          </button>
        )}
      </div>
      <div className="term-host" ref={host} />
    </div>
  );
}
