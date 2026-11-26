import { http } from "../../http";
import { parseTriage } from "./parse";

export const triageApi = {
  decide: (message: string) => http.post("/triage", { message }, parseTriage),
};
