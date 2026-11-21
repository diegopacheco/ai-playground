import { addDays, startOfMonth } from "../../time";
import type { Range } from "./types";

export type Period = { readonly from: Date; readonly to: Date };

export function periodFor(range: Range, weekStart: Date): Period {
  switch (range) {
    case "week":
      return { from: weekStart, to: addDays(weekStart, 7) };
    case "month": {
      const from = startOfMonth(weekStart);
      return { from, to: new Date(from.getFullYear(), from.getMonth() + 1, 1) };
    }
    case "all":
      return { from: new Date(2000, 0, 1), to: new Date(2100, 0, 1) };
    default: {
      const unreachable: never = range;
      return unreachable;
    }
  }
}
