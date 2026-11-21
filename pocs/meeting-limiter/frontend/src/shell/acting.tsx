import { createContext, use, useEffect, useState, type ReactNode } from "react";
import { setUser } from "../logger";
import { readJson, writeJson } from "./storage";

const STORAGE_KEY = "ml.actingUser";
const DEFAULT_USER = "gabriela";

type Acting = { readonly userId: string; readonly setUserId: (id: string) => void };

const ActingContext = createContext<Acting | undefined>(undefined);

export function ActingUserProvider({ children }: { readonly children: ReactNode }) {
  const [userId, setUserId] = useState(() => readJson(STORAGE_KEY, (v) => (typeof v === "string" ? v : undefined), DEFAULT_USER));
  useEffect(() => {
    writeJson(STORAGE_KEY, userId);
    setUser(userId);
  }, [userId]);
  return <ActingContext value={{ userId, setUserId }}>{children}</ActingContext>;
}

export function useActingUser(): Acting {
  const ctx = use(ActingContext);
  if (!ctx) throw new Error("ActingUserProvider is missing");
  return ctx;
}
