import type { ActorState } from "../types";

type StateBadgeProps = { readonly state: ActorState };

export function StateBadge({ state }: StateBadgeProps) {
  return <span className={`badge state-${state.toLowerCase()}`}>{state}</span>;
}
