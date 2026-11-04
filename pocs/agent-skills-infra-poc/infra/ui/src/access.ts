import { createContext, useContext } from "react";
import type { Session } from "./types";

export type Capability = "state" | "logs" | "evals";

export type Access = {
  session: Session;
  app: string;
  setApp: (app: string) => void;
};

export const AccessContext = createContext<Access | null>(null);

export function useAccess(): Access {
  const access = useContext(AccessContext);
  if (!access) throw new Error("AccessContext missing");
  return access;
}

export function readsAll(scopes: string[], capability: Capability): boolean {
  return scopes.includes(`${capability}:read:all`);
}

export function readsOwn(scopes: string[], capability: Capability): boolean {
  return scopes.includes(`${capability}:read`);
}

export function isAdmin(scopes: string[]): boolean {
  return scopes.includes("clients:read");
}

export function viewOf(scopes: string[], capability: Capability): "all" | "own" | "none" {
  if (readsAll(scopes, capability)) return "all";
  if (readsOwn(scopes, capability)) return "own";
  return "none";
}
