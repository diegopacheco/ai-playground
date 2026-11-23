import { ApiError, http } from "../../http.ts";
import type { Report } from "./types.ts";

function parseReport(raw: unknown): Report {
  if (typeof raw === "object" && raw !== null && "samples" in raw && Array.isArray(raw.samples) && "rules" in raw && Array.isArray(raw.rules)) {
    // SAFETY: the shared server type defines this payload and the top level fields were checked above
    return raw as Report;
  }
  throw new ApiError(422, "invalid report payload");
}

export const reportApi = {
  get: () => http.get("/report", parseReport),
};
