function required(name: string, value: string | undefined): string {
  if (!value) throw new Error(`${name} is required`);
  return value;
}

export const config = {
  serviceName: "anti-slop-api",
  port: Number(required("API_PORT", Bun.env.API_PORT)),
  rootDir: new URL("..", import.meta.url).pathname,
  slopDir: "slop",
  oxlintConfig: "oxlint.config.ts",
} as const;
