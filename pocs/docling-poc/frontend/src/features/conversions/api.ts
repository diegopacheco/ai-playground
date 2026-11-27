import { http } from "../../http";
import { parseConversion, parseExtensions, parseHistory } from "./types";

export type UploadInput = { readonly file: File; readonly ocr: boolean };

export const conversionsApi = {
  history: () => http.get("/api/conversions", parseHistory),
  get: (id: string) => http.get(`/api/conversions/${encodeURIComponent(id)}`, parseConversion),
  formats: () => http.get("/api/formats", parseExtensions),
  upload: ({ file, ocr }: UploadInput) =>
    http.upload(
      `/api/conversions?ocr=${ocr}`,
      file,
      { "X-Filename": encodeURIComponent(file.name), "Content-Type": "application/octet-stream" },
      parseConversion,
    ),
};
