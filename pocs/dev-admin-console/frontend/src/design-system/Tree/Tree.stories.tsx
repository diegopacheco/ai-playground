import type { Meta, StoryObj } from "@storybook/react-vite";
import { Tree } from "./Tree";

const meta: Meta<typeof Tree> = {
  title: "Design System/Tree",
  component: Tree
};

export default meta;
type Story = StoryObj<typeof Tree>;

export const SqlSchema: Story = {
  args: {
    nodes: [
      {
        name: "customers",
        kind: "table",
        detail: "BASE TABLE",
        children: [
          { name: "id", kind: "column", detail: "integer not null" },
          { name: "email", kind: "column", detail: "character varying not null" },
          { name: "country", kind: "column", detail: "character not null" }
        ]
      },
      { name: "order_totals", kind: "view", detail: "VIEW" }
    ]
  }
};

export const RedisKeys: Story = {
  args: {
    nodes: [
      { name: "config:app:name", kind: "string", detail: "string" },
      {
        name: "session:abc123",
        kind: "hash",
        detail: "4 fields",
        children: [
          { name: "user", kind: "field", detail: "diego" },
          { name: "ip", kind: "field", detail: "10.0.0.7" }
        ]
      },
      { name: "leaderboard", kind: "zset", detail: "4 members" }
    ]
  }
};

export const EtcdPrefixes: Story = {
  args: {
    nodes: [
      {
        name: "config",
        kind: "prefix",
        detail: "3 keys",
        children: [
          {
            name: "app",
            kind: "prefix",
            detail: "2 keys",
            children: [{ name: "name", kind: "key", detail: "dev-admin-console" }]
          }
        ]
      }
    ]
  }
};

export const Empty: Story = {
  args: { nodes: [], emptyLabel: "no tables in this schema" }
};
