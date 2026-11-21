import type { Cents } from "../../money";
import type { User } from "../people/types";
import type { Meeting } from "../meetings/types";

export type Spender = {
  readonly user: User;
  readonly spent: Cents;
  readonly minutes: number;
  readonly meetings: number;
};

export type Rankings = {
  readonly meetings: readonly Meeting[];
  readonly spenders: readonly Spender[];
  readonly timeSinks: readonly Spender[];
};

export const RANGES = ["week", "month", "all"] as const;
export type Range = (typeof RANGES)[number];

export function rangeLabel(r: Range): string {
  switch (r) {
    case "week":
      return "This week";
    case "month":
      return "This month";
    case "all":
      return "All time";
    default: {
      const unreachable: never = r;
      return unreachable;
    }
  }
}
