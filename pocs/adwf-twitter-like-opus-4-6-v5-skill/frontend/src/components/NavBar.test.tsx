import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { AuthContext } from "../context/AuthContext";
import { NavBar } from "./NavBar";

function renderNavBar(isAuthenticated: boolean) {
  const authValue = {
    user: isAuthenticated ? { id: 1, username: "test", email: "t@t.com", display_name: "Test User", bio: "", created_at: "" } : null,
    token: isAuthenticated ? "fake-token" : null,
    setAuth: vi.fn(),
    logout: vi.fn(),
    isAuthenticated,
  };

  return render(
    <AuthContext.Provider value={authValue}>
      <MemoryRouter>
        <NavBar />
      </MemoryRouter>
    </AuthContext.Provider>
  );
}

describe("NavBar", () => {
  it("renders nothing when not authenticated", () => {
    const { container } = renderNavBar(false);
    expect(container.innerHTML).toBe("");
  });

  it("renders Home link when authenticated", () => {
    renderNavBar(true);
    expect(screen.getByText("Home")).toBeInTheDocument();
  });

  it("renders Profile link when authenticated", () => {
    renderNavBar(true);
    expect(screen.getByText("Profile")).toBeInTheDocument();
  });

  it("renders Chirper brand link", () => {
    renderNavBar(true);
    expect(screen.getByText("Chirper")).toBeInTheDocument();
  });

  it("renders Logout button when authenticated", () => {
    renderNavBar(true);
    expect(screen.getByText("Logout")).toBeInTheDocument();
  });
});
