export const config = {
  appName: "agent-substrate-poc-ui",
  apiUrl: import.meta.env.VITE_API_URL ?? "",
  refreshMs: 2000,
} as const;
