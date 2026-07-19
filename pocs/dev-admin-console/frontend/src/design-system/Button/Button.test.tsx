import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button } from "./Button";

describe("Button", () => {
  it("calls its handler when clicked", async () => {
    const onClick = jest.fn();
    render(<Button onClick={onClick}>Run</Button>);
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("does not call its handler while disabled, so an in-flight query cannot be fired twice", async () => {
    const onClick = jest.fn();
    render(<Button onClick={onClick} disabled>Run</Button>);
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(onClick).not.toHaveBeenCalled();
  });

  it("marks the variant so destructive actions are visually distinct from safe ones", () => {
    const { rerender } = render(<Button variant="danger">Delete</Button>);
    expect(screen.getByRole("button")).toHaveClass("ds-button-danger");
    rerender(<Button variant="primary">Save</Button>);
    expect(screen.getByRole("button")).toHaveClass("ds-button-primary");
  });

  it("defaults to the secondary variant so a forgotten prop is never destructive-looking", () => {
    render(<Button>Cancel</Button>);
    expect(screen.getByRole("button")).toHaveClass("ds-button-secondary");
  });
});
