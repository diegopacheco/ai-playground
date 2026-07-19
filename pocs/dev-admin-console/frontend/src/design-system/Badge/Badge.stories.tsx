import type { Meta, StoryObj } from "@storybook/react-vite";
import { Badge } from "./Badge";

const meta: Meta<typeof Badge> = {
  title: "Design System/Badge",
  component: Badge,
  args: { children: "postgres" }
};

export default meta;
type Story = StoryObj<typeof Badge>;

export const Neutral: Story = {};
export const Accent: Story = { args: { tone: "accent" } };
export const Allowed: Story = { args: { tone: "ok", children: "allowed" } };
export const Denied: Story = { args: { tone: "error", children: "denied" } };

export const KeyTypes: Story = {
  render: () => (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {["string", "hash", "list", "set", "zset", "stream", "topic", "index", "prefix"].map((kind) => (
        <Badge key={kind}>{kind}</Badge>
      ))}
    </div>
  )
};
