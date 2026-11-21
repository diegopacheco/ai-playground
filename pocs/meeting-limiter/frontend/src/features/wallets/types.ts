import type { Cents } from "../../money";
import type { User } from "../people/types";

export type Wallet = {
  readonly user: User;
  readonly budget: Cents;
  readonly spent: Cents;
  readonly balance: Cents;
  readonly minutesUsed: number;
  readonly minutesLimit: number;
  readonly meetings: number;
};

export type WalletHealth = "healthy" | "low" | "empty";

export function walletHealth(w: Pick<Wallet, "budget" | "balance">): WalletHealth {
  if (w.balance <= 0) return "empty";
  return w.balance / w.budget < 0.25 ? "low" : "healthy";
}
