import type { Status } from "../types";

type StatusBadgeProps = { readonly status: Status };

export function StatusBadge({ status }: StatusBadgeProps) {
  return <span className={`badge badge-${status}`}>{status}</span>;
}
