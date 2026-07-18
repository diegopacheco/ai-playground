import type { Preview } from "@storybook/react-vite";
import "../src/styles/global.css";

const preview: Preview = {
  parameters: {
    backgrounds: {
      default: "console",
      values: [{ name: "console", value: "#FAF5EE" }]
    },
    controls: { matchers: { color: /(background|color)$/i } }
  }
};

export default preview;
