import type { Meta, StoryObj } from "@storybook/react-vite";
import { DataGrid } from "./DataGrid";

const meta: Meta<typeof DataGrid> = {
  title: "Design System/DataGrid",
  component: DataGrid
};

export default meta;
type Story = StoryObj<typeof DataGrid>;

const columns = ["id", "email", "full_name", "country", "deleted_at"];
const rows = Array.from({ length: 12 }, (_, index) => ({
  id: String(index + 1),
  email: `customer${index + 1}@example.com`,
  full_name: `Customer ${index + 1}`,
  country: ["BR", "US", "DE", "JP", "PT"][index % 5],
  deleted_at: index % 4 === 0 ? null : "2026-01-01"
}));

export const Rows: Story = { args: { columns, rows } };

export const WithRowActivation: Story = {
  args: { columns, rows, onRowActivate: (index: number) => window.alert(`row ${index + 1}`) }
};

export const NoRows: Story = { args: { columns, rows: [], emptyLabel: "0 rows returned" } };

export const NothingRunYet: Story = {
  args: { columns: [], rows: [], emptyLabel: "run a statement to see results" }
};
