type Level = "info" | "warn" | "error";
type Fields = Readonly<Record<string, string | number | boolean | undefined>>;

export type Logger = {
  readonly info: (event: string, fields?: Fields) => void;
  readonly warn: (event: string, fields?: Fields) => void;
  readonly error: (event: string, fields?: Fields) => void;
};

export function createLogger(service: string): Logger {
  const write = (level: Level, event: string, fields: Fields = {}) => {
    const entry = { ts: new Date().toISOString(), level, service, event, ...fields };
    process.stderr.write(JSON.stringify(entry) + "\n");
  };
  return {
    info: (event, fields) => write("info", event, fields),
    warn: (event, fields) => write("warn", event, fields),
    error: (event, fields) => write("error", event, fields),
  };
}
