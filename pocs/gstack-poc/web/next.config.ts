import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@qa2pw/runner"],
  experimental: {
    serverComponentsExternalPackages: ["playwright"],
  },
};

export default config;
