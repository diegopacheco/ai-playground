type AppConfig = {
  readonly appName: string;
  readonly defaultLighting: "day" | "golden" | "night";
  readonly islandSize: number;
};

function readLighting(raw: string | undefined): "day" | "golden" | "night" {
  if (raw === "golden" || raw === "night") {
    return raw;
  }
  return "day";
}

export const config: AppConfig = {
  appName: import.meta.env.VITE_APP_NAME ?? "sf-pixel-city",
  defaultLighting: readLighting(import.meta.env.VITE_DEFAULT_LIGHTING),
  islandSize: 24,
} as const;
