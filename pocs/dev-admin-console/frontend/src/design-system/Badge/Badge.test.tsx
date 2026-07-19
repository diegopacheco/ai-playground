import { render, screen } from "@testing-library/react";
import { Badge } from "./Badge";

describe("Badge", () => {
  it("renders its label", () => {
    render(<Badge>hash</Badge>);
    expect(screen.getByText("hash")).toBeInTheDocument();
  });

  it("uses the error tone for denials so a rejected write is visually unmistakable", () => {
    render(<Badge tone="error">denied</Badge>);
    expect(screen.getByText("denied")).toHaveClass("ds-badge-error");
  });

  it("defaults to a neutral tone so an unstyled badge never implies success or failure", () => {
    render(<Badge>string</Badge>);
    expect(screen.getByText("string")).toHaveClass("ds-badge-neutral");
  });
});
