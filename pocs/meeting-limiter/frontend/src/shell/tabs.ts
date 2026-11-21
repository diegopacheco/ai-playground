import type { IconName } from "./icons";

export type TabPath = "/" | "/meetings" | "/wallets" | "/rankings";

export type Tab = {
  readonly id: string;
  readonly label: string;
  readonly path: TabPath;
  readonly icon: IconName;
  readonly digit: string;
};

export const TABS: readonly Tab[] = [
  { id: "calendar", label: "Calendar", path: "/", icon: "calendar", digit: "1" },
  { id: "meetings", label: "Meetings", path: "/meetings", icon: "list", digit: "2" },
  { id: "wallets", label: "Wallets", path: "/wallets", icon: "wallet", digit: "3" },
  { id: "rankings", label: "Rankings", path: "/rankings", icon: "trophy", digit: "4" },
];
