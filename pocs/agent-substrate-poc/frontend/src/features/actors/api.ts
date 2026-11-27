import { http } from "../../http";
import type { Action } from "./rules";
import { parseActor, parseActors, parseAgentState, parseDeleted, parseNote, parseWorkers } from "./types";

const path = (name: string) => `/api/actors/${encodeURIComponent(name)}`;

export const actorsApi = {
  list: () => http.get("/api/actors", parseActors),
  get: (name: string) => http.get(path(name), parseActor),
  create: (name: string) => http.post("/api/actors", { name }, parseActor),
  state: (name: string) => http.get(`${path(name)}/state`, parseAgentState),
  remember: (name: string, text: string) => http.post(`${path(name)}/notes`, { text }, parseNote),
  act: async (name: string, action: Action): Promise<void> => {
    if (action === "delete") {
      await http.delete(path(name), parseDeleted);
      return;
    }
    await http.post(`${path(name)}/${action}`, {}, parseActor);
  },
  workers: () => http.get("/api/workers", parseWorkers),
};
