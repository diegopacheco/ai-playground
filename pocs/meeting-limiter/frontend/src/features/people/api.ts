import { list, num, obj, str } from "../../decode";
import { ApiError, http } from "../../http";
import { ROLES, type Policy, type Role, type User } from "./types";

function parseRole(raw: string): Role {
  const role = ROLES.find((r) => r === raw);
  if (!role) throw new ApiError(422, `unknown role ${raw}`, null);
  return role;
}

export function parseUser(raw: unknown): User {
  const o = obj(raw, "user");
  return { id: str(o, "id"), name: str(o, "name"), title: str(o, "title"), role: parseRole(str(o, "role")) };
}

function parsePolicy(raw: unknown): Policy {
  const o = obj(raw, "policy");
  return {
    role: parseRole(str(o, "role")),
    weeklyHours: num(o, "weeklyHours"),
    hourlyRate: num(o, "hourlyRate"),
    weeklyBudget: num(o, "weeklyBudget"),
  };
}

export const peopleApi = {
  users: () => http.get("/users", (raw) => list(raw, parseUser, "users")),
  policies: () => http.get("/policies", (raw) => list(raw, parsePolicy, "policies")),
};
