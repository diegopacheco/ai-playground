import type { ConnectionKind } from "@lib/types";
import "./EngineLogo.css";

export interface EngineLogoProps {
  kind: ConnectionKind;
  size?: number;
}

const BRAND: Record<ConnectionKind, string> = {
  postgres: "#336791",
  mysql: "#00758F",
  cassandra: "#1287B1",
  redis: "#DC382D",
  etcd: "#419EDA",
  kafka: "#3B3838",
  elasticsearch: "#C2A200"
};

function glyph(kind: ConnectionKind) {
  switch (kind) {
    case "postgres":
      return (
        <>
          <ellipse cx="12" cy="13" rx="7" ry="6.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <path d="M8 8 Q12 3 16 8" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <circle cx="9.6" cy="12" r="1.1" fill="currentColor" />
          <circle cx="14.4" cy="12" r="1.1" fill="currentColor" />
          <path d="M12 15.5 L12 19" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </>
      );
    case "mysql":
      return (
        <>
          <path d="M4 15 Q9 6 15 8 Q20 10 19 17" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <path d="M19 17 L16.5 15.5 M19 17 L20.5 14.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <circle cx="13.5" cy="10.5" r="1.1" fill="currentColor" />
        </>
      );
    case "cassandra":
      return (
        <>
          <path d="M3 12 Q12 4 21 12 Q12 20 3 12 Z" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <circle cx="12" cy="12" r="2.8" fill="currentColor" />
        </>
      );
    case "redis":
      return (
        <>
          <path d="M4 8 L12 5 L20 8 L12 11 Z" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
          <path d="M4 12 L12 15 L20 12" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
          <path d="M4 16 L12 19 L20 16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
        </>
      );
    case "etcd":
      return (
        <>
          <path d="M12 3 L20 7.5 L20 16.5 L12 21 L4 16.5 L4 7.5 Z" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
          <path d="M12 3 L12 12 L4 7.5 M12 12 L20 7.5 M12 12 L12 21" fill="none" stroke="currentColor" strokeWidth="1.3" opacity="0.75" />
        </>
      );
    case "kafka":
      return (
        <>
          <circle cx="7" cy="6.5" r="2.4" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <circle cx="7" cy="17.5" r="2.4" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <circle cx="17" cy="12" r="2.8" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <path d="M9.2 7.6 L14.4 10.8 M9.2 16.4 L14.4 13.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </>
      );
    case "elasticsearch":
      return (
        <>
          <circle cx="10.5" cy="10.5" r="6" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <path d="M15 15 L20.5 20.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <path d="M7.5 10.5 L13.5 10.5 M8.5 8 L12.5 8 M8.5 13 L12.5 13" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
        </>
      );
  }
}

export function EngineLogo({ kind, size = 24 }: EngineLogoProps) {
  return (
    <span
      className="ds-engine-logo"
      style={{ color: BRAND[kind], width: size, height: size }}
      data-testid={`engine-logo-${kind}`}
      aria-hidden="true"
    >
      <svg viewBox="0 0 24 24" width={size} height={size} role="presentation">
        {glyph(kind)}
      </svg>
    </span>
  );
}

export const engineBrand = BRAND;
