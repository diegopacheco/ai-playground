import type { NextConfig } from "next";

const config: NextConfig = {
  serverExternalPackages: ["@fly-zord/agents"],
  devIndicators: false
};

export default config;
