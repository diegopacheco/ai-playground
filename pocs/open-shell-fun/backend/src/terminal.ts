import type { ServerWebSocket, Subprocess } from "bun";

export type TerminalData = { name: string; proc?: Subprocess };

export type ClientMessage =
  | { type: "input"; data: string }
  | { type: "resize"; cols: number; rows: number };

export function parseClientMessage(raw: string): ClientMessage | null {
  try {
    const msg = JSON.parse(raw) as Record<string, unknown>;
    if (msg.type === "input" && typeof msg.data === "string") {
      return { type: "input", data: msg.data };
    }
    const cols = Number(msg.cols);
    const rows = Number(msg.rows);
    if (msg.type === "resize" && Number.isInteger(cols) && Number.isInteger(rows) && cols > 0 && rows > 0 && cols <= 500 && rows <= 200) {
      return { type: "resize", cols, rows };
    }
    return null;
  } catch {
    return null;
  }
}

export function shellArgs(name: string): string[] {
  return ["openshell", "sandbox", "exec", name, "--tty", "--", "bash", "-l"];
}

export const terminalSocket = {
  open(ws: ServerWebSocket<TerminalData>) {
    const proc = Bun.spawn(shellArgs(ws.data.name), {
      env: { ...process.env, TERM: "xterm-256color" },
      terminal: {
        cols: 120,
        rows: 32,
        data: (_term, bytes) => ws.send(bytes),
      },
    });
    ws.data.proc = proc;
    void proc.exited.then((code) => ws.close(1000, `shell exited ${code}`));
  },
  message(ws: ServerWebSocket<TerminalData>, raw: string | Buffer) {
    const msg = parseClientMessage(raw.toString());
    const term = ws.data.proc?.terminal;
    if (!msg || !term) {
      return;
    }
    if (msg.type === "input") {
      term.write(msg.data);
    } else {
      term.resize(msg.cols, msg.rows);
    }
  },
  close(ws: ServerWebSocket<TerminalData>) {
    ws.data.proc?.kill();
  },
};
