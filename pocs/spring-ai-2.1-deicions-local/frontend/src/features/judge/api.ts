import { http } from "../../http";
import { parseVerdict } from "./parse";

export const judgeApi = {
  judge: (question: string, answer: string) => http.post("/judge", { question, answer }, parseVerdict),
};
