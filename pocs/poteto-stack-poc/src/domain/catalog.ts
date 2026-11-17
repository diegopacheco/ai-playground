import data from "../catalog/pstack.json";
import type { Catalog, Playbook } from "./types";

export const catalog = data as Catalog;

export function findPlaybook(id: string): Playbook | undefined {
  return catalog.playbooks.find((p) => p.id === id);
}
