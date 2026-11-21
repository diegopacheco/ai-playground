import type { Cents } from "../../money";

export const ROLES = ["engineer", "manager", "director", "vp"] as const;
export type Role = (typeof ROLES)[number];

export type User = {
  readonly id: string;
  readonly name: string;
  readonly title: string;
  readonly role: Role;
};

export type Policy = {
  readonly role: Role;
  readonly weeklyHours: number;
  readonly hourlyRate: Cents;
  readonly weeklyBudget: Cents;
};

export function roleLabel(role: Role): string {
  switch (role) {
    case "engineer":
      return "Engineer";
    case "manager":
      return "Manager";
    case "director":
      return "Director";
    case "vp":
      return "VP";
    default: {
      const unreachable: never = role;
      return unreachable;
    }
  }
}

export function initials(name: string): string {
  return name
    .split(" ")
    .map((p) => p[0] ?? "")
    .join("")
    .slice(0, 2)
    .toUpperCase();
}
