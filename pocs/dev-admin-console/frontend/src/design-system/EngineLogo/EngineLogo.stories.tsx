import type { Meta, StoryObj } from "@storybook/react-vite";
import { EngineLogo } from "./EngineLogo";
import type { ConnectionKind } from "@lib/types";

const KINDS: ConnectionKind[] = [
  "postgres", "mysql", "cassandra", "redis", "etcd", "kafka", "elasticsearch"
];

const meta: Meta<typeof EngineLogo> = {
  title: "Design System/EngineLogo",
  component: EngineLogo,
  args: { kind: "postgres", size: 30 }
};

export default meta;
type Story = StoryObj<typeof EngineLogo>;

export const Single: Story = {};

export const AllEngines: Story = {
  render: () => (
    <div style={{ display: "flex", gap: 22, alignItems: "center", flexWrap: "wrap" }}>
      {KINDS.map((kind) => (
        <div key={kind} style={{ textAlign: "center", fontSize: 11 }}>
          <EngineLogo kind={kind} size={34} />
          <div style={{ marginTop: 6, color: "#97806F" }}>{kind}</div>
        </div>
      ))}
    </div>
  )
};
