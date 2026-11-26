import { http } from "../../http";
import { parseScreening } from "./parse";
import type { Battery } from "./types";

export const guardrailApi = {
  screen: (battery: Battery, text: string) => http.post("/guardrail", { battery, text }, parseScreening),
};
