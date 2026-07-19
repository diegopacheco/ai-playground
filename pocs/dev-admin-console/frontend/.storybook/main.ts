import type { StorybookConfig } from "@storybook/react-vite";
import { fileURLToPath } from "node:url";

const config: StorybookConfig = {
  stories: ["../src/**/*.stories.@(ts|tsx)"],
  framework: { name: "@storybook/react-vite", options: {} },
  viteFinal: async (viteConfig) => {
    viteConfig.resolve = viteConfig.resolve ?? {};
    viteConfig.resolve.alias = {
      ...viteConfig.resolve.alias,
      "@design": fileURLToPath(new URL("../src/design-system", import.meta.url)),
      "@console": fileURLToPath(new URL("../src/console", import.meta.url)),
      "@engines": fileURLToPath(new URL("../src/engines", import.meta.url)),
      "@lib": fileURLToPath(new URL("../src/lib", import.meta.url))
    };
    return viteConfig;
  }
};

export default config;
