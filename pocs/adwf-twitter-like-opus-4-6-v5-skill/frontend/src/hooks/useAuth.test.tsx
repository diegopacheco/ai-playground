import { renderHook } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { AuthContext } from "../context/AuthContext";
import { useAuth } from "./useAuth";
import type { ReactNode } from "react";

function createWrapper(token: string | null) {
  return function Wrapper({ children }: { children: ReactNode }) {
    const value = {
      user: token ? { id: 1, username: "testuser", email: "test@test.com", display_name: "Test", bio: "", created_at: "" } : null,
      token,
      setAuth: vi.fn(),
      logout: vi.fn(),
      isAuthenticated: !!token,
    };
    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
  };
}

describe("useAuth", () => {
  it("returns token when provided", () => {
    const { result } = renderHook(() => useAuth(), { wrapper: createWrapper("my-token") });
    expect(result.current.token).toBe("my-token");
  });

  it("returns null token when not authenticated", () => {
    const { result } = renderHook(() => useAuth(), { wrapper: createWrapper(null) });
    expect(result.current.token).toBeNull();
  });

  it("returns isAuthenticated true when token exists", () => {
    const { result } = renderHook(() => useAuth(), { wrapper: createWrapper("abc") });
    expect(result.current.isAuthenticated).toBe(true);
  });

  it("returns isAuthenticated false when no token", () => {
    const { result } = renderHook(() => useAuth(), { wrapper: createWrapper(null) });
    expect(result.current.isAuthenticated).toBe(false);
  });

  it("returns user when authenticated", () => {
    const { result } = renderHook(() => useAuth(), { wrapper: createWrapper("tok") });
    expect(result.current.user).not.toBeNull();
    expect(result.current.user?.username).toBe("testuser");
  });
});
