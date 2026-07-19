import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Tree, type TreeNode } from "./Tree";

const nodes: TreeNode[] = [
  {
    name: "customers",
    kind: "table",
    detail: "BASE TABLE",
    children: [
      { name: "id", kind: "column", detail: "integer not null" },
      { name: "email", kind: "column", detail: "character varying not null" }
    ]
  },
  { name: "order_totals", kind: "view", detail: "VIEW" }
];

describe("Tree", () => {
  it("hides columns until the table is expanded, so a wide schema stays readable", () => {
    render(<Tree nodes={nodes} />);
    expect(screen.getByText("customers")).toBeInTheDocument();
    expect(screen.queryByText("email")).not.toBeInTheDocument();
  });

  it("reveals columns when the table is expanded", async () => {
    render(<Tree nodes={nodes} />);
    await userEvent.click(screen.getByRole("button", { name: "Expand customers" }));
    expect(screen.getByText("email")).toBeInTheDocument();
    expect(screen.getByText("character varying not null")).toBeInTheDocument();
  });

  it("collapses again so an operator can fold away a table they are done with", async () => {
    render(<Tree nodes={nodes} />);
    await userEvent.click(screen.getByRole("button", { name: "Expand customers" }));
    await userEvent.click(screen.getByRole("button", { name: "Collapse customers" }));
    expect(screen.queryByText("email")).not.toBeInTheDocument();
  });

  it("reports the full path when a node is selected, so the editor can insert a qualified name", async () => {
    const onSelect = jest.fn();
    render(<Tree nodes={nodes} onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: "Expand customers" }));
    await userEvent.click(screen.getByText("email"));
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ name: "email" }),
      ["customers", "email"]
    );
  });

  it("does not offer an expander for leaf nodes", () => {
    render(<Tree nodes={nodes} />);
    expect(screen.queryByRole("button", { name: /Expand order_totals/ })).not.toBeInTheDocument();
  });

  it("shows the kind badge so redis key types and table types are visible without expanding", () => {
    render(<Tree nodes={nodes} />);
    expect(screen.getByText("table")).toBeInTheDocument();
    expect(screen.getByText("view")).toBeInTheDocument();
  });

  it("shows an empty message rather than a blank panel when a connection has no objects", () => {
    render(<Tree nodes={[]} emptyLabel="no tables" />);
    expect(screen.getByText("no tables")).toBeInTheDocument();
  });

  it("renders deeply nested prefixes, which is how etcd keys arrive", async () => {
    const etcd: TreeNode[] = [
      { name: "config", kind: "prefix", children: [
        { name: "app", kind: "prefix", children: [{ name: "name", kind: "key", detail: "admin-console" }] }
      ] }
    ];
    render(<Tree nodes={etcd} />);
    await userEvent.click(screen.getByRole("button", { name: "Expand config" }));
    await userEvent.click(screen.getByRole("button", { name: "Expand app" }));
    expect(screen.getByText("admin-console")).toBeInTheDocument();
  });
});
